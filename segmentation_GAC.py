import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

def segmentation_GAC(df_error_w, iterations_seg):

    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """
    
        def _store(x):
            lst.append(np.copy(x))
    
        return _store
    
    
    # Morphological ACWE
    image = img_as_float(df_error_w)
    
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 50, init_level_set=init_ls,
                                 smoothing=3, iter_callback=callback)
      

    
    '''
    fig6,ax6 = plt.subplots(figsize=(6,6))
    ax6.imshow(image, cmap="gray")
    ax6.set_axis_off()
    ax6.contour(ls, [0.5], colors='r')
    ax6.set_title("Morphological GAC segmentation", fontsize=12)
    
    ax6.imshow(ls, cmap="gray")
    ax6.set_axis_off()
    contour = ax6.contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax6.contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax6.contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration " + str(iterations_seg))
    ax6.legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax6.set_title(title, fontsize=12)
    
    fig6.tight_layout()
    plt.show()
    '''
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    ax = axes.flatten()
    
    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("ACWE Segmentierung", fontsize=10)
    
    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[10], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 10")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 50")
    ax[1].legend(loc="upper right")
    title = "ACWE Evolution"
    ax[1].set_title(title, fontsize=10)
    
    '''
    # Morphological GAC
    image = img_as_float(data.coins())
    gimage = inverse_gaussian_gradient(image)
    
    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, num_iter=230,
                                               init_level_set=init_ls,
                                               smoothing=1, balloon=-1,
                                               threshold=0.69,
                                               iter_callback=callback)
    '''
    # Morphological GAC
    image = img_as_float(df_error_w)
    gimage = inverse_gaussian_gradient(image)
    #plt.imshow(gimage)
    
    # Initial level set
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[4:-4, 4:-4] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 500,
                                               init_level_set=init_ls,
                                               smoothing=1, balloon=-1,
                                               threshold=0.13,
                                               iter_callback=callback)
    
    df_holes_GAC = pd.DataFrame(ls)
    
    
    
    ax[2].imshow(image, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("GAC Segmentierung", fontsize=10)
    
    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[200], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 200")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 500")
    ax[3].legend(loc="upper right")
    title = "GAC Evolution"
    ax[3].set_title(title, fontsize=10)
    
    fig.tight_layout()
    plt.savefig('ACWEundGAC.png', dpi=300)
    plt.show()
    
    return df_holes_GAC