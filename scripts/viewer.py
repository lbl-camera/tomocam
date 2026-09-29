
import numpy as np
import pyqtgraph as pg
import click
import h5py


def load_hdf5(file_path, dataset_name="recon"):
    """Load data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

def create_image_viewer(data):
    """Create an image viewer using PyQtGraph."""
    app = pg.mkQApp("CT Viewer")
    
    # Create an ImageView widget directly
    img_view = pg.ImageView()
    img_view.setWindowTitle('CT Image Viewer')
    img_view.resize(800, 800)
    img_view.show()

    # Set the data to the ImageView
    img_view.setImage(data)

    # Start the Qt event loop
    pg.exec()

@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--dataset-name', default='recon', help='Name of the dataset to load from the HDF5 file.')
def plot(file_path, dataset_name):
    """Load data from HDF5 and display it in an image viewer."""
    data = load_hdf5(file_path, dataset_name)
    create_image_viewer(data)


if __name__ == "__main__":
    plot()
