#!/usr/bin/env python
"""MBIR reconstruction driver for raw DXchange (/exchange) beamline files.

Standalone python driver; it does not read the JSON that build/mbir_recon uses.

    python run_mbir.py raw.h5 --axis 1227                    # all sinograms
    python run_mbir.py raw.h5 --sino 1000:1032 --axis 1227
    python run_mbir.py raw.h5 --sino 1000:1032 --axis 1227 --save-prep prep.h5
    python run_mbir.py prep.h5 --from-prep --axis 1227 --iters 50
    python run_mbir.py --config recon.json          # flags override the file

Pipeline: read /exchange/{data,data_white,data_dark,theta} -> flat/dark
normalize -> clip -> minus_log -> stripe removal (Fourier-wavelet) -> MBIR ->
multi-page TIFF (one float32 page per slice, <out-dir>/<stem>.tiff).

`--save-prep PATH` keeps the preprocessed float32 sinograms (`projs`, shape
(nproj, nrow, ncol)) and radian angles (`angs`) so a later run can start from
them with `--from-prep`, skipping the normalization and stripe removal.

--config takes a flat JSON object whose keys are the long flag names with
underscores, plus "input" for the file, e.g. {"input": "raw.h5",
"sino": "1000:1032", "axis": 1227, "iters": 50, "save_prep": "prep.h5"}.
`--write-template recon.json` writes a file with every key and its default.
Slices are START:STOP[:STEP], or a JSON list [START, STOP[, STEP]].

MPI: launch with srun/mpirun and the --sino rows are split into contiguous
slabs, one per rank (needs mpi4py and a tomocam built with MULTI_PROC). Each
rank reads and preprocesses only its slab; rank 0 gathers, masks and writes
the result. --save-prep/--prep-only are single-process only.
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import h5py
import numpy as np
import tifffile

try:
    from mpi4py import MPI
    RANK = MPI.COMM_WORLD.Get_rank()
    SIZE = MPI.COMM_WORLD.Get_size()
    # Every rank starting as a singleton means mpi4py was built against a
    # different MPI than the one srun launches (each would redo all slices).
    if SIZE == 1 and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        sys.exit(f"mpi4py sees 1 rank but Slurm started {os.environ['SLURM_NTASKS']}: "
                 "rebuild mpi4py against Cray MPICH")
except ImportError:
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        sys.exit("run under MPI needs mpi4py, which is not installed")
    RANK, SIZE = 0, 1

# mpi4py first: it initializes MPI, which tomocam then reuses.
import tomopy
import tomocam

PREP_DATASET = "projs"
PREP_ANGLES = "angs"


class SliceType(click.ParamType):
    """START:STOP[:STEP] (or a JSON list of those ints) as a slice."""
    name = "slice"

    def convert(self, value, param, ctx):
        if isinstance(value, slice):
            return value
        try:
            parts = (value.split(":") if isinstance(value, str) else list(value))
            nums = [int(x) for x in parts]
        except (TypeError, ValueError):
            self.fail(f"{value!r} is not START:STOP[:STEP]", param, ctx)
        if len(nums) not in (2, 3):
            self.fail(f"{value!r} is not START:STOP[:STEP]", param, ctx)
        start, stop = nums[:2]
        step = nums[2] if len(nums) == 3 else 1
        if start < 0 or stop <= start or step < 1:
            self.fail(f"{value!r} needs 0 <= START < STOP and STEP >= 1",
                      param, ctx)
        return slice(start, stop, step)


SLICE = SliceType()

# config-file key -> click parameter name, where they differ
CONFIG_ALIAS = {"input": "input_file", "format": "fmt"}
# values for options with no default, so the template shows what goes there
TEMPLATE_EXAMPLES = {"input_file": "raw.h5", "sino": "0:16", "axis": 1024.0}
NOT_IN_TEMPLATE = {"config", "write_template"}


def row_indices(sl, nrow):
    """Detector row numbers a slice selects, for naming the output files."""
    return np.arange(*sl.indices(nrow))


def split_rows(sino, nrow, rank, size):
    """All rows `sino` selects, and a slice covering this rank's contiguous
    share of them. The shares differ by at most one row."""
    rows = row_indices(sino, nrow)
    if rows.size < size:
        sys.exit(f"{rows.size} slices cannot be split across {size} ranks")
    mine = np.array_split(rows, size)[rank]
    return rows, slice(int(mine[0]), int(mine[-1]) + 1, sino.step or 1)


def load_raw(path, sino, proj, part=(0, 1)):
    """Read this rank's share of the selected rows of a raw DXchange file:
    tomo, flat, dark, theta, and the row numbers of all ranks together."""
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    with h5py.File(path, "r") as h:
        if "exchange" not in h:
            sys.exit(f"{path} has no /exchange group -- not a DXchange file")
        ex = h["exchange"]
        for key in ("data", "data_white", "data_dark"):
            if key not in ex:
                sys.exit(f"/exchange/{key} missing from {path}")
        nrow = ex["data"].shape[1]
        if sino.stop is not None and sino.stop > nrow:
            sys.exit(f"--sino {sino.start}:{sino.stop} is outside the {nrow} "
                     f"detector rows of {path.name}")
        rows, mine = split_rows(sino, nrow, *part)
        tomo = ex["data"][proj, mine, :]
        flat = ex["data_white"][:, mine, :]
        dark = ex["data_dark"][:, mine, :]
        theta = ex["theta"][proj] if "theta" in ex else None
    if tomo.size == 0:
        sys.exit(f"--sino/--proj select nothing from {path.name}")
    return tomo, flat, dark, theta, rows


def preprocess(tomo, flat, dark, theta, clip_min, do_stripe):
    """Normalize, clip, take -log, remove stripes. Returns float32 tomo and
    radian theta."""
    if theta is None:
        theta = np.linspace(0.0, 180.0, tomo.shape[0], dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    # these files store degrees; the reconstruction takes radians
    if theta.max() > 2.0 * np.pi:
        theta = np.deg2rad(theta)

    tomo = tomo.astype(np.float32)
    tomo = tomopy.normalize(tomo, flat, dark, out=tomo)
    np.nan_to_num(tomo, copy=False, nan=clip_min, posinf=clip_min,
                  neginf=clip_min)
    tomo[tomo < clip_min] = clip_min
    tomo = tomopy.minus_log(tomo)
    if do_stripe:
        tomo = tomopy.remove_stripe_fw(tomo)
    return np.ascontiguousarray(tomo, dtype=np.float32), theta


def save_prep_file(path, tomo, theta, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        d = h.create_dataset(PREP_DATASET, data=tomo)
        for k, v in attrs.items():
            d.attrs[k] = v
        a = h.create_dataset(PREP_ANGLES, data=theta)
        a.attrs["units"] = "radians"
    print(f"wrote prepped data {path} {tomo.shape} float32")


def load_prep(path, sino, part=(0, 1)):
    """Read this rank's share of a file written by save_prep, optionally a
    subset of its rows. Also returns the row numbers of all ranks together."""
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    with h5py.File(path, "r") as h:
        for key in (PREP_DATASET, PREP_ANGLES):
            if key not in h:
                sys.exit(f"dataset '{key}' not in {path} -- was it written "
                         f"with --save-prep?")
        dset = h[PREP_DATASET]
        if dset.ndim != 3:
            sys.exit(f"'{PREP_DATASET}' must be 3-D, got {dset.ndim}-D")
        nrow = dset.shape[1]
        if sino.stop is not None and sino.stop > nrow:
            sys.exit(f"--sino {sino.start}:{sino.stop} is outside the {nrow} "
                     f"rows of {path.name}")
        rows, mine = split_rows(sino, nrow, *part)
        # name slices by their detector row in the raw file, when recorded
        src = dset.attrs.get("source_rows")
        if src is not None and len(src) == nrow:
            rows = np.asarray(src)[rows]
        tomo = dset[:, mine, :]
        theta = h[PREP_ANGLES][:]
    theta = np.asarray(theta, dtype=np.float32).ravel()
    if theta.size != tomo.shape[0]:
        sys.exit(f"'{PREP_ANGLES}' has {theta.size} entries but "
                 f"'{PREP_DATASET}' has {tomo.shape[0]} projections")
    if theta.max() > 2.0 * np.pi:
        print("warning: angles look like degrees, converting to radians")
        theta = np.deg2rad(theta)
    return np.ascontiguousarray(tomo, dtype=np.float32), theta, rows


def to_sinogram_order(tomo):
    """(nproj, nrow, ncol) -> (nslice, nproj, ncol), as tomocam.recon wants.

    An even-width sinogram loses its last column, matching mbir_recon, so the
    center of rotation keeps its meaning across the python and C++ drivers.
    """
    if tomo.shape[2] % 2 == 0:
        tomo = tomo[:, :, :-1]
    return np.ascontiguousarray(np.transpose(tomo, (1, 0, 2)))


def write_recon(rec, rows, out_dir, stem, fmt):
    """One multi-page TIFF (a page per slice), or one HDF5 'recon' dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "hdf5":
        outfile = out_dir / f"{stem}.h5"
        with h5py.File(outfile, "w") as h:
            d = h.create_dataset("recon", data=rec)
            d.attrs["rows"] = rows
        print(f"wrote {outfile} {rec.shape} float32")
    else:
        outfile = out_dir / f"{stem}.tiff"
        # one IFD per slice so every viewer sees a real multi-page file
        tifffile.imwrite(outfile, rec.astype('f'))
        print(f"wrote {outfile} {rec.shape} float32, one page per slice")


def load_config(ctx, param, value):
    """Eager --config callback: a flat JSON object becomes the option defaults."""
    if value is None:
        return None
    try:
        with open(value) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise click.BadParameter(f"cannot read {value}: {e}")
    if not isinstance(cfg, dict):
        raise click.BadParameter("must hold a flat JSON object")
    cfg = {CONFIG_ALIAS.get(k, k).replace("-", "_"): v for k, v in cfg.items()}
    known = {p.name for p in ctx.command.params} - {"config"}
    unknown = sorted(set(cfg) - known)
    if unknown:
        click.echo(f"warning: ignoring unknown --config keys {unknown}",
                   err=True)
    ctx.default_map = {k: v for k, v in cfg.items() if k in known}
    return value


def write_template(ctx, param, value):
    """Eager --write-template callback: dump every option's default as JSON."""
    if value is None:
        return None
    to_key = {v: k for k, v in CONFIG_ALIAS.items()}
    tmpl = {}
    for p in ctx.command.params:
        if p.name in NOT_IN_TEMPLATE:
            continue
        val = TEMPLATE_EXAMPLES.get(p.name, p.default)
        if isinstance(val, Path):
            val = str(val)
        elif getattr(p, "is_flag", False):
            val = False
        elif not isinstance(val, (bool, int, float, str)):
            val = None  # no default (click may hand back a sentinel here)
        tmpl[to_key.get(p.name, p.name)] = val
    text = json.dumps(tmpl, indent=4) + "\n"
    if str(value) == "-":
        click.echo(text, nl=False)
    else:
        if value.exists():
            raise click.BadParameter(f"{value} exists, not overwriting")
        value.parent.mkdir(parents=True, exist_ok=True)
        value.write_text(text)
        click.echo(f"wrote config template {value}")
    ctx.exit()


@click.command(context_settings={"show_default": True},
               help=__doc__.split("\n\n", 1)[0])
@click.argument("input_file", metavar="[INPUT]", required=False,
                type=click.Path(path_type=Path))
@click.option("--config", type=click.Path(dir_okay=False), is_eager=True,
              expose_value=False, callback=load_config,
              help="Flat JSON of option defaults; flags override it.")
@click.option("--write-template", type=click.Path(dir_okay=False, path_type=Path),
              is_eager=True, expose_value=False, callback=write_template,
              metavar="PATH.json",
              help="Write a JSON config template with every option and exit "
                   "('-' for stdout).")
@click.option("--sino", type=SLICE, metavar="START:STOP[:STEP]",
              help="Detector rows to use [default: all rows].")
@click.option("--proj", type=SLICE, metavar="START:STOP[:STEP]",
              help="Projections to use [default: all].")
@click.option("--axis", type=float,
              help="Center of rotation, in pixels of the raw detector.")
@click.option("--clip-min", type=float, default=0.01,
              help="Floor applied before minus_log.")
@click.option("--no-stripe", is_flag=True,
              help="Skip Fourier-wavelet stripe removal.")
@click.option("--save-prep", type=click.Path(dir_okay=False, path_type=Path),
              metavar="PATH.h5",
              help="Save the preprocessed sinograms and angles here.")
@click.option("--from-prep", is_flag=True,
              help="INPUT is a prepped file; skip preprocessing.")
@click.option("--prep-only", is_flag=True,
              help="Stop after --save-prep; do not reconstruct.")
@click.option("--iters", type=click.IntRange(min=1), default=100,
              help="Max MBIR iterations.")
@click.option("--sigma", type=click.FloatRange(min=0, min_open=True),
              default=1000.0, help="qGGMRF sigma.")
@click.option("--tol", type=float, default=1e-4, help="Cost tolerance.")
@click.option("--xtol", type=float, default=1e-4, help="Update tolerance.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              help="Output directory [default: ./<input stem>_recon].")
@click.option("--format", "fmt", type=click.Choice(["tiff", "hdf5"]),
              default="tiff",
              help="tiff: multi-page <out-dir>/<stem>.tiff; hdf5: <out-dir>/<stem>.h5.")
@click.option("--preview", is_flag=True,
              help="Also write a PNG of the middle slice.")
@click.option("--show", is_flag=True,
              help="Display the preview interactively.")
def main(input_file, sino, proj, axis, clip_min, no_stripe, save_prep,
         from_prep, prep_only, iters, sigma, tol, xtol, out_dir, fmt,
         preview, show):
    if input_file is None:
        raise click.UsageError("INPUT file required (argument or in --config)")
    if axis is None:
        raise click.UsageError("--axis is required")
    if SIZE > 1 and save_prep:
        raise click.UsageError("--save-prep/--prep-only are single-process; "
                               "run them without MPI")
    if from_prep and save_prep:
        raise click.UsageError("--save-prep has no effect with --from-prep")
    if prep_only and not save_prep:
        raise click.UsageError("--prep-only needs --save-prep")
    sino = sino or slice(None)
    proj = proj or slice(None)

    def say(msg):
        if RANK == 0:
            click.echo(msg)

    part = (RANK, SIZE)
    stem = input_file.stem
    t_start = time.time()

    if from_prep:
        say("reading prepped data ...")
        tomo, theta, rows = load_prep(input_file, sino, part)
    else:
        say("== prep")
        say(f"raw file:    {input_file}")
        say(f"sino / proj: {sino} / {proj}   ranks: {SIZE}")
        t0 = time.time()
        tomo, flat, dark, theta, rows = load_raw(input_file, sino, proj, part)
        say(f"read {tomo.shape} in {time.time() - t0:.1f}s (rank 0's slab)")
        t0 = time.time()
        tomo, theta = preprocess(tomo, flat, dark, theta,
                                 clip_min, not no_stripe)
        say(f"preprocessing {time.time() - t0:.1f}s")
        if save_prep:
            save_prep_file(save_prep, tomo, theta, {
                "source_file": str(input_file),
                "source_rows": rows,
                "clip_min": clip_min,
                "remove_stripe": not no_stripe,
            })
        if prep_only:
            say(f"total {time.time() - t_start:.1f}s")
            return

    tomo = to_sinogram_order(tomo)
    ncol = tomo.shape[2]
    if not 0.0 <= axis < ncol:
        raise click.UsageError(
            f"axis={axis} is outside the detector width ({ncol} px)")

    out_dir = out_dir or Path.cwd() / f"{stem}_recon"

    say("== recon")
    say(f"sinogram:  {tomo.shape} per rank, {rows.size} slices "
        f"(rows {rows[0]}..{rows[-1]}) on {SIZE} rank(s)")
    say(f"axis:      {axis}")
    say(f"iters:     {iters}")
    say(f"sigma:     {sigma}")
    say(f"tol/xtol:  {tol} {xtol}")
    say(f"output:    {out_dir} ({fmt})")

    t0 = time.time()
    # tomocam.recon takes smoothness = 1/sigma
    rec = tomocam.recon_mpi(tomo, theta, axis, num_iters=iters,
                        smoothness=1.0 / sigma, tol=tol, xtol=xtol)
    say(f"MBIR {time.time() - t0:.1f}s")

    # the gathered volume lives on rank 0 only; the others hold an empty array.
    # Rank 0 tells the rest whether the gather worked, so a failure exits every
    # rank instead of leaving them waiting at the barrier below.
    ok = RANK != 0 or rec.shape[0] == rows.size
    if SIZE > 1:
        ok = MPI.COMM_WORLD.bcast(ok, root=0)
    if not ok:
        sys.exit(f"gathered {rec.shape[0]} slices, expected {rows.size} -- was "
                 f"tomocam built with MULTI_PROC?" if RANK == 0 else 1)

    if RANK == 0:
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.99)

        write_recon(rec, rows, out_dir, stem, fmt)

        if preview or show:
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mid = rec[rec.shape[0] // 2]
            lo, hi = np.percentile(mid, [1, 99])
            plt.imshow(mid, cmap="Greys_r", vmin=lo, vmax=hi)
            plt.title(f"{stem}  slice {rows[rec.shape[0] // 2]}  axis={axis}")
            plt.colorbar()
            png = out_dir / f"{stem}_preview.png"
            plt.savefig(png, dpi=150, bbox_inches="tight")
            click.echo(f"wrote {png}")
            if show:
                plt.show()

        click.echo(f"total {time.time() - t_start:.1f}s")

    if SIZE > 1:
        MPI.COMM_WORLD.Barrier()


if __name__ == '__main__':
    main()
