#!/usr/bin/env python
"""MBIR reconstruction driver -- the python twin of build/mbir_recon.

Usage: python run_mbir.py <input.json> [--skip-prep] [--bregman] [--show]

Reads the same JSON that mbir_recon.cpp consumes (src/config.h), so one file
describes a dataset for both. The top-level keys name an HDF5 file that already
holds float32 sinograms and radian angles:

    filename     HDF5 file to reconstruct from                   (required)
    dataset      projection dataset, (nproj, nrow, ncol) float32 (required)
    angles       angles dataset, radians, float32                (required)
    axis         center of rotation, in pixels                   (required)
    slices       [start, stop] detector rows of `dataset` to reconstruct
                                             (default: the whole stack)
    MBIR.max_iters                                           (default: 100)
    MBIR.inner_iters  CG inner-loop cap, split-Bregman only     (default: 1)
    MBIR.tol                                                (default: 1e-4)
    MBIR.xtol                                               (default: 1e-4)
    MBIR.mu       split-Bregman penalty weight, bregman only   (default: 10.)
    MBIR.lambda   TV shrinkage weight, bregman only            (default: 0.1)
    MBIR.sigma    qGGMRF parameter                             (default: 500)
    output.filename  where the reconstruction is written        (required)
    output.format    "hdf5" or "tiff"      (default: from the filename suffix)

Raw APS 32-ID / DXchange files hold uint16 counts and degrees, which this
pipeline cannot read. If the JSON carries a `prep` block, it is normalized into
the file named by `filename` first (--skip-prep reuses an existing one):

    prep.input_dir      directory holding the raw file            (required)
    prep.input_file     raw APS 32-ID / DXchange HDF5 file        (required)
    prep.sino           [start, stop] or [start, stop, step] detector rows
                        to extract, in raw-file indexing          (required)
    prep.proj           [start, stop] or [start, stop, step] projections
                                                            (default: all)
    prep.clip_min       floor applied before minus_log         (default: 0.01)
    prep.remove_stripe  Fourier-wavelet stripe removal         (default: true)

Note that `slices` then indexes the prepped file, not the raw one: [0, nrow].
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile

import tomopy
import tomocam

# ReconParams defaults, src/common.h -- keep in step with the C++ struct
DEFAULTS = {
    "max_iters": 100,
    "inner_iters": 1,
    "tol": 1.0e-4,
    "xtol": 1.0e-4,
    "mu": 10.0,
    "lambda": 0.1,
    "sigma": 500.0,
}


def as_slice(spec, name):
    """Turn a [start, stop] or [start, stop, step] list into a slice."""
    if spec is None:
        return slice(None)
    if not isinstance(spec, (list, tuple)) or len(spec) not in (2, 3):
        raise ValueError(
            f"'{name}' must be [start, stop] or [start, stop, step], got {spec!r}")
    start, stop = int(spec[0]), int(spec[1])
    step = int(spec[2]) if len(spec) == 3 else 1
    if stop <= start or step < 1:
        raise ValueError(
            f"'{name}' must have stop > start and step >= 1, got {spec!r}")
    return slice(start, stop, step)


def load_recon_params(cfg):
    """The MBIR block, with the defaults load_recon_params() leaves in place."""
    mbir = cfg.get("MBIR", {})
    unknown = set(mbir) - set(DEFAULTS)
    if unknown:
        print(f"warning: ignoring unknown MBIR keys {sorted(unknown)}")
    params = dict(DEFAULTS)
    params.update({k: v for k, v in mbir.items() if k in DEFAULTS})
    params["max_iters"] = int(params["max_iters"])
    params["inner_iters"] = int(params["inner_iters"])
    for key in ("tol", "xtol", "mu", "lambda", "sigma"):
        params[key] = float(params[key])
    if params["max_iters"] < 1:
        sys.exit("'MBIR.max_iters' must be greater than 0")
    if params["sigma"] <= 0:
        sys.exit("'MBIR.sigma' must be greater than 0")
    return params


def load_output_params(cfg):
    """The output block. Mirrors load_output_params(): suffix picks the format."""
    if "output" not in cfg or "filename" not in cfg.get("output", {}):
        sys.exit("'output.filename' missing from the JSON file")
    out = cfg["output"]
    outfile = Path(out["filename"])
    fmt = out.get("format")
    if fmt is None:
        fmt = "tiff" if outfile.suffix in (".tif", ".tiff") else "hdf5"
    if fmt not in ("hdf5", "tiff"):
        sys.exit(f"'output.format' must be \"hdf5\" or \"tiff\", got {fmt!r}")
    return outfile, fmt


def prep(cfg, json_file):
    """Normalize a raw APS 32-ID file into the float32 file `filename` names.

    mbir_recon reads `dataset` straight out of the HDF5 file with no flat/dark
    normalization, rejects anything that is not float32, and feeds `angles`
    directly into cos/sin, so they must be radians. This does the tomopy half
    of the pipeline once and writes the result where the rest of the JSON --
    and the C++ binary reading the same file -- expects to find it.
    """
    p = cfg["prep"]
    for key in ("input_dir", "input_file", "sino"):
        if key not in p:
            sys.exit(f"'prep.{key}' missing from {json_file}")

    raw = Path(p["input_dir"]) / p["input_file"]
    if not raw.exists():
        raise FileNotFoundError(f"File {raw} not found")

    sino = as_slice(p["sino"], "prep.sino")
    proj = as_slice(p.get("proj"), "prep.proj")
    clip_min = float(p.get("clip_min", 0.01))
    do_stripe = bool(p.get("remove_stripe", True))

    outfile = Path(cfg["filename"])
    sino_name = cfg.get("dataset", "sino")
    angs_name = cfg.get("angles", "theta")

    print("== prep")
    print("raw file:   ", raw)
    print("prep file:  ", outfile)
    print("sino:       ", p["sino"])
    print("proj:       ", p.get("proj"))

    t_start = time.time()
    with h5py.File(raw, "r") as h:
        exchange = h["exchange"]
        tomo = exchange["data"][proj, sino, :]
        flat = exchange["data_white"][:, sino, :]
        dark = exchange["data_dark"][:, sino, :]
        theta = exchange["theta"][proj] if "theta" in exchange else None
    if tomo.size == 0:
        sys.exit(f"prep.sino={p['sino']} selects no rows; check the detector "
                 f"height of {raw.name}")
    print(f"read {tomo.shape} in {time.time() - t_start:.1f}s")

    if theta is None:
        theta = np.linspace(0.0, 180.0, tomo.shape[0], dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)

    # these files store degrees; the reconstruction takes the angles in radians
    if theta.max() > 2.0 * np.pi:
        theta = np.deg2rad(theta)

    t0 = time.time()
    tomo = tomo.astype(np.float32)
    tomo = tomopy.normalize(tomo, flat, dark, out=tomo)
    np.nan_to_num(tomo, copy=False, nan=clip_min, posinf=clip_min,
                  neginf=clip_min)
    tomo[tomo < clip_min] = clip_min
    tomo = tomopy.minus_log(tomo)
    if do_stripe:
        tomo = tomopy.remove_stripe_fw(tomo)
    tomo = np.ascontiguousarray(tomo, dtype=np.float32)
    print(f"preprocessing {time.time() - t0:.1f}s")

    # h5::Reader::read_sinogram slices dim 1, so keep projection order
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outfile, "w") as h:
        d = h.create_dataset(sino_name, data=tomo)
        d.attrs["source_file"] = str(raw)
        d.attrs["source_rows"] = np.asarray(p["sino"][:2], dtype=np.int64)
        d.attrs["clip_min"] = clip_min
        d.attrs["remove_stripe"] = do_stripe
        a = h.create_dataset(angs_name, data=theta)
        a.attrs["units"] = "radians"
    nproj, nrow, ncol = tomo.shape
    print(f"wrote {outfile} {tomo.shape} float32, {nproj} angles")
    print(f'set "slices": [0, {nrow}] to reconstruct all of it; '
          f"prep took {time.time() - t_start:.1f}s")


def read_sinogram(cfg, dataset, angles):
    """Read the slice range named by `slices` in sinogram order.

    The file holds (nproj, nrow, ncol); tomocam wants (nslice, nproj, ncol),
    which is the transpose h5::Reader::read_sinogram does on the C++ side.
    """
    filename = Path(cfg["filename"])
    if not filename.exists():
        raise FileNotFoundError(
            f"File {filename} not found -- run the prep step, or point "
            f"'filename' at an existing prepped file")

    with h5py.File(filename, "r") as h:
        if dataset not in h:
            sys.exit(f"dataset '{dataset}' not in {filename}")
        if angles not in h:
            sys.exit(f"dataset '{angles}' not in {filename}")
        dset = h[dataset]
        if dset.ndim != 3:
            sys.exit(f"'{dataset}' must be 3-D, got {dset.ndim}-D")

        nrow = dset.shape[1]
        ibeg, iend = 0, nrow
        if "slices" in cfg:
            slcs = as_slice(cfg["slices"], "slices")
            ibeg, iend = slcs.start, slcs.stop
            if slcs.step != 1:
                print(f"warning: 'slices' step {slcs.step} ignored, "
                      f"mbir_recon reads a contiguous range")
            if iend > nrow:
                sys.exit(f"slices={cfg['slices']} is outside the {nrow} rows "
                         f"of '{dataset}' in {filename}")

        tomo = dset[:, ibeg:iend, :]
        theta = h[angles][:]

    if tomo.dtype != np.float32:
        # the C++ reader throws "Data type mismatch" here rather than convert
        print(f"warning: '{dataset}' is {tomo.dtype}, not float32 -- "
              f"mbir_recon would reject this file")
    tomo = np.ascontiguousarray(tomo, dtype=np.float32)
    theta = np.ascontiguousarray(theta, dtype=np.float32).ravel()

    if theta.size != tomo.shape[0]:
        sys.exit(f"'{angles}' has {theta.size} entries but '{dataset}' has "
                 f"{tomo.shape[0]} projections")
    if theta.max() > 2.0 * np.pi:
        print("warning: angles look like degrees, converting to radians")
        theta = np.deg2rad(theta)

    # mbir_recon drops the last column of an even-width sinogram and leaves the
    # center where it is; do the same so both paths reconstruct the same volume
    if tomo.shape[2] % 2 == 0:
        tomo = tomo[:, :, :-1]

    # (nproj, nslice, ncol) -> (nslice, nproj, ncol)
    return np.ascontiguousarray(np.transpose(tomo, (1, 0, 2))), theta, ibeg


def write_recon(rec, outfile, fmt):
    """Write the volume as one HDF5 'recon' dataset, or one TIFF per slice."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "hdf5":
        with h5py.File(outfile, "w") as h:
            h.create_dataset("recon", data=rec)
        print(f"wrote {outfile} {rec.shape} float32")
    else:
        stem = outfile.parent / outfile.stem
        for i, image in enumerate(rec):
            tifffile.imwrite(f"{stem}_{i:05d}.tiff",
                             np.asarray(image, dtype=np.float32))
        print(f"wrote {rec.shape[0]} slices to {stem}_*.tiff")


def main():
    parser = argparse.ArgumentParser(
        description="MBIR reconstruction from an mbir_recon JSON config")
    parser.add_argument("json_file", help="input JSON file")
    parser.add_argument("--skip-prep", action="store_true",
                        help="reconstruct from an existing prep file without "
                             "regenerating it")
    parser.add_argument("--preview", action="store_true",
                        help="also write a PNG of the middle slice")
    parser.add_argument("--show", action="store_true",
                        help="display the preview interactively")
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        cfg = json.load(f)

    for key in ("filename", "dataset", "angles", "axis"):
        if key not in cfg:
            sys.exit(f"'{key}' missing from {args.json_file}")

    dataset = cfg["dataset"]
    angles = cfg["angles"]
    params = load_recon_params(cfg)
    outfile, fmt = load_output_params(cfg)

    # tomocam.recon is the mbir2 path; the split-Bregman keys belong to
    # build/mbir_bregman_recon, which reads this same JSON
    bregman_keys = sorted(set(cfg.get("MBIR", {})) & {"inner_iters", "mu", "lambda"})
    if bregman_keys:
        print(f"note: MBIR {bregman_keys} apply to build/mbir_bregman_recon; "
              f"this script runs the mbir2 path and ignores them")

    t_start = time.time()
    if "prep" in cfg and not args.skip_prep:
        prep(cfg, args.json_file)

    tomo, theta, ibeg = read_sinogram(cfg, dataset, angles)

    ncol = tomo.shape[2]
    axis = float(cfg["axis"])
    if not 0.0 <= axis < ncol:
        sys.exit(f"axis={axis} is outside the detector width ({ncol} px)")

    print("== recon")
    print("data file:  ", cfg["filename"])
    print("sinogram:   ", tomo.shape, f"(rows {ibeg}..{ibeg + tomo.shape[0]})")
    print("axis:       ", axis)
    print("max_iters:  ", params["max_iters"])
    print("sigma:      ", params["sigma"])
    print("tol:        ", params["tol"])
    print("xtol:       ", params["xtol"])
    print("output:     ", outfile, f"({fmt})")

    t0 = time.time()
    # tomocam.recon takes smoothness = 1/sigma, mbir_recon takes sigma directly
    rec = tomocam.recon(tomo, theta, axis,
                        num_iters=params["max_iters"],
                        smoothness=1.0 / params["sigma"],
                        tol=params["tol"], xtol=params["xtol"])
    rec = tomopy.circ_mask(rec, axis=0, ratio=1.0)
    print(f"MBIR {time.time() - t0:.1f}s")

    write_recon(rec, outfile, fmt)

    if args.preview or args.show:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mid = rec[rec.shape[0] // 2]
        lo, hi = np.percentile(mid, [1, 99])
        plt.imshow(mid, cmap="Greys_r", vmin=lo, vmax=hi)
        plt.title(f"{outfile.stem}  slice {ibeg + rec.shape[0] // 2}  "
                  f"axis={axis}")
        plt.colorbar()
        preview = outfile.parent / f"{outfile.stem}_preview.png"
        plt.savefig(preview, dpi=150, bbox_inches="tight")
        print(f"wrote {preview}")
        if args.show:
            plt.show()

    print(f"total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
