# A tomocam MBIR reconstruction pipeline as podman container entrypoint

from pathlib import Path
import numpy as np
import tomopy
import h5py
import tomocam
import tifffile
from mpi4py import MPI
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
        else:
            tomo = dset[:]
        
        flat = f['exchange/data_white'][:,ibegin:iend, :]
        dark = f['exchange/data_dark'][:,ibegin:iend, :]
        theta = f['exchange/theta'][:]
        if np.any(theta > 2 * np.pi):
            theta = theta * np.pi / 180.0
    
    return tomo, flat, dark, theta


def tomocam_pipeline(datadir, filename, axis, num_iters=50, smoothness=0.01, tol=1e-5, xtol=1e-5):
    """ Tomocam MBIR reconstruction pipeline. Partitions data across MPI ranks along the rotation axis.

    Parameters:
    datadir : str
        Directory where the data file is located
    filename : str
        Name of the data file
    axis : float
        Center of rotation
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
    
    # Init MPI
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
            logger.info(f"Input: {datadir}/{filename}, Rotation axis: {axis}")
        
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

        # Get the size of data on rank 0 and broadcast
        if rank == 0:
            with h5py.File(dataset, 'r') as f:
                dset = f['exchange/data']
                num_projs, height, width = dset.shape
            logger.info(f"Data shape: {num_projs} projections x {height} slices x {width} pixels")
        else:
            num_projs, height, width = None, None, None
        
        num_projs, height, width = comm.bcast((num_projs, height, width), root=0)

        # Calculate dynamic slice distribution
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
        
        logger.info(f"Processing slices {ibegin} to {iend} ({my_share} slices)")

        # Load data from file
        tomo, flat, dark, theta = read_als_832h5(dataset, sino=(ibegin, iend))
        tomo = tomo.astype(np.float32)
        theta = theta.astype(np.float32)
        
        logger.info("Data loaded, starting normalization")
        tomo = tomopy.normalize(tomo, flat, dark, out=tomo)

        # Apply threshold and preprocessing
        tomo[tomo < 0.01] = 0.01
        tomo = tomopy.minus_log(tomo)
        tomo = tomopy.remove_stripe_fw(tomo)

        tomo = np.transpose(tomo, (1, 0, 2))
        
        logger.info("Starting MBIR reconstruction")

        # MBIR reconstruction, it gathers data from all ranks internally
        rec = tomocam.recon_mpi(tomo, theta, center=axis, num_iters=num_iters, smoothness=smoothness, tol=tol, xtol=xtol)
        rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

        # Save reconstructed data with input filename base
        input_base = Path(filename).stem
        outfile = outdir / f'{input_base}_recon.tif'
        
        if rank == 0:
            logger.info(f"Saving reconstruction to {outfile}")
        
        tifffile.imwrite(outfile, rec.astype(np.float32), imagej=True)
        
        if rank == 0:
            logger.info("Reconstruction complete")
    
    except Exception as e:
        logger.error(f"Error during reconstruction: {str(e)}", exc_info=True)
        comm.Abort(1)
        sys.exit(1)

@click.command()
@click.option('--datadir', type=str, required=True, help='Directory where the data file is located')
@click.option('--filename', type=str, required=True, help='Name of the data file')
@click.option('--axis', type=float, required=True, help='Center of rotation')
@click.option('--num_iters', type=int, default=50, help='Number of MBIR iterations')
@click.option('--smoothness', type=float, default=0.01, help='Smoothness parameter for MBIR')
@click.option('--tol', type=float, default=1e-5, help='Tolerance for MBIR convergence')
@click.option('--xtol', type=float, default=1e-5, help='Tolerance for MBIR convergence')
def main(datadir, filename, axis, num_iters, smoothness, tol, xtol):
    try:
        tomocam_pipeline(datadir, filename, axis, num_iters, smoothness, tol, xtol)
    finally:
        MPI.Finalize()


if __name__ == '__main__':
    main()
