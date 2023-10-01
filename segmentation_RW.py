import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

def segmentation_RW(df_error_w):

    data = df_error_w

    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < -15] = 1
    markers[data > -3] = 2

    # Run random walker algorithm
    labels = random_walker(data, markers, beta=0, mode='bf')
    
    df_holes_RW = pd.DataFrame(labels)
    df_holes_RW = df_holes_RW.replace(2, 0)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2.4), sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Daten', fontsize = 10)
    ax2.imshow(markers, cmap='magma')
    ax2.axis('off')
    ax2.set_title('Marker', fontsize = 10)
    ax3.imshow(labels, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Segmentierung', fontsize = 10)

    fig.tight_layout()
    plt.savefig('RandomWalker.png', dpi=300)
    plt.show()
    
    
    return df_holes_RW
    