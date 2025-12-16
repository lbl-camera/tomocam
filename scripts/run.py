# A tomocam MBIR reconstruction pipeline as podman container entrypoint

from pathlib import Path
import numpy as np
import tomopy
import h5py
import tomocam
import tifffile
from mpi4py import MPI
from datetime import datetime
import click
import sys
import logging




def setup_logging():
    """Setup MPI-aware logging"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(levelname)s: %(message)s',
        stream=sys.stdout
    )
    return logging.getLogger(__name__)


def validate_volumes():
    """Validate that required volume mounts exist"""
    input_vol = Path('/data/input')
    output_vol = Path('/data/output')
    
    if not input_vol.exists():
        raise RuntimeError(f"Input volume {input_vol} not mounted")
    if not output_vol.exists():
        raise RuntimeError(f"Output volume {output_vol} not mounted")


def read_als_832h5(filename, sino=None):
    """Read ALS 8.3.2 HDF5 format using h5py"""
    with h5py.File(filename, 'r') as f:
        dset = f['exchange/data']
        if sino is not None:
            ibegin, iend = sino
            tomo = dset[:, ibegin:iend, :]
            flat = f['exchange/data_white'][:, ibegin:iend, :]
            dark = f['exchange/data_dark'][:, ibegin:iend, :]
        else:
            tomo = dset[:]
            flat = f['exchange/data_white'][:]
            dark = f['exchange/data_dark'][:]
        
        theta = f['exchange/theta'][:]
        if np.any(theta > 2 * np.pi):
            theta = theta * np.pi / 180.0
    
    return tomo, flat, dark, theta


def tomocam_pipeline(datadir, filename, axis=None, num_iters=50, smoothness=0.01, tol=1e-5, xtol=1e-5):
    """ Tomocam MBIR reconstruction pipeline. Partitions data across MPI ranks 

    Parameters:
    filename : str
        Name of the data file
    axis: float, default None
        Center of rotation in pixels
    num_iters : int, default 50
        Number of MBIR iterations
    smoothness : float, default 0.01
        Smoothness parameter for MBIR
    tol : float, default 1e-5
        Tolerance for MBIR convergence
    xtol : float, default 1e-5
        Tolerance for MBIR convergence
    Returns:
    None
    """
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # FOR DEBUGGING ONLY
    if not size == 4:
        raise RuntimeError("This script is configured to run with 4 MPI ranks only for debugging")
    
    logger = setup_logging()
    
    try:
        # Validate volumes on rank 0
        if rank == 0:
            validate_volumes()
            logger.info(f"Starting reconstruction with {size} MPI ranks")
        
        # Sync all ranks after validation
        comm.Barrier()
        
        # /data/input is mounted podman volume for input data
        input_data = Path('/data/input') / datadir
        output_data = Path('/data/output') / datadir
        
        if rank == 0:
            if not output_data.exists():
                output_data.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_data}")
        
        comm.Barrier()
        outdir = output_data

        dataset = input_data / filename
        
        # Validate dataset exists on all ranks
        if not dataset.exists():
            raise FileNotFoundError(f"File {dataset} not found")

        # First (size-1) ranks get slices as multiples of 16, last rank gets remainder
        """ FIX THIS LATER 
        if size > 1:
            base_slices = (height // 16 // (size - 1)) * 16
            
            if rank < size - 1:
                my_share = base_slices
                ibegin = rank * base_slices
            else:
                # Last rank gets all remaining slices
                my_share = height - (size - 1) * base_slices
                ibegin = (size - 1) * base_slices
        else:
            # Single rank gets all slices
            my_share = height
            ibegin = 0
        """ 

        # FOR DEBUGGING ONLY: Equal slice distribution 
        base_slices = 64
        ibegin = rank * base_slices
        iend = ibegin + base_slices
        
        logger.info(f"Processing slices {ibegin} to {iend}: ({base_slices} slices)")

        # Load data from file
        tomo, flat, dark, theta = read_als_832h5(filename)
        tomo = tomo.astype(np.float32)
        theta = theta.astype(np.float32)

        # ensure theta is in radians
        if np.any(theta > 2 * np.pi):
            theta = theta * np.pi / 180.0
        
        logger.info("Data loaded, starting normalization")
        tomo = tomopy.normalize(tomo, flat, dark, out=tomo)

        logger.info("Normalization complete, starting center-of-rotation search")
        # rank 0 only for COR search
        # Center of rotation search
        # find projection at 0 and 180 degrees
        if axis is None:
            if rank == 0:
                proj0 = tomo[0,:,:].copy()
                t180 = theta[0] + np.pi
                idx180 = (np.abs(theta - t180)).argmin()
                proj180 = tomo[idx180,:,:].copy()
                cor = tomopy.find_center_pc(proj0, proj180, tol=0.5)
                logger.info(f"Center of rotation found at: {cor:.2f}")
            else:
                cor = None

            # Broadcast COR to all ranks
            axis = comm.bcast(cor, root=0)

        # Apply threshold and preprocessing
        tomo[tomo < 0.01] = 0.01
        tomo = tomopy.minus_log(tomo)
        tomo = tomopy.remove_stripe_fw(tomo, level=7, sigma=3)

        # Transpose data to (slices, projections, pixels)
        sino = np.transpose(tomo, (1, 0, 2))
        logger.info("Starting MBIR reconstruction")

        my_sino = sino[ibegin:iend, :, :].copy()
        # MBIR reconstruction, it gathers data from all ranks internally
        rec = tomocam.recon_mpi(my_sino, theta, center=cor, num_iters=num_iters, smoothness=smoothness, tol=tol, xtol=xtol)
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.98)

        # Save reconstructed data with input filename base
        outdir = Path('/data/output')
        outfile = outdir / output_file
        
        if rank == 0:
            logger.info(f"Saving reconstruction to {outfile}")
            tifffile.imwrite(outfile, rec.astype(np.float32), imagej=True)
        
        logger.info("Reconstruction complete")

    except Exception as e:
        logger.error(f"Error during reconstruction: {str(e)}", exc_info=True)
        comm.Abort(1)
        sys.exit(1)

@click.command()
@click.option('--filename', type=str, required=True, help='Name of the data file')
@click.option('--output-file', type=str, default='recon.tif', help='Output filename for the reconstruction')
@click.option('--axis', type=float, default=None, help='Center of rotation')
@click.option('--num-iters', type=int, default=50, help='Number of MBIR iterations')
@click.option('--smoothness', type=float, default=0.01, help='Smoothness parameter for MBIR')
@click.option('--tol', type=float, default=1e-5, help='Tolerance for MBIR convergence')
@click.option('--xtol', type=float, default=1e-5, help='Tolerance for MBIR convergence')

def main(filename, output_file, axis, num_iters, smoothness, tol, xtol):
    try:
        MPI.Init()
        validate_volumes()
        # ensure filename is a valid  path
        filename = Path('/data/input') / filename
        if not filename.exists():
            raise FileNotFoundError(f"Input file {filename} not found")
        tomocam_pipeline(filename, output_file, axis, num_iters, smoothness, tol, xtol)
        
    finally:
        MPI.Finalize()


if __name__ == '__main__':
    main()
