def main(filename, path_to_file, path_export, n_resize, show, file_format, photo):
    
    #%% start timer
    import time
    start_time = time.time()
    
    # %% import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import sys #for organizing paths
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Patch
    import plotstyle as pltst #import custom plotstyle
    import pandas as pd
    #x_lim=(100,650)
    #y_lim=()
    
    # %% import data
    import importdata as im
    
    df, info, magnification, units_conversion, data_size = im.import_data(path_to_file, n_resize) # import data as pandas dataframes
    #determine field of view from imported data
    fov = [0,float(info.iloc[8,1])*float(info.iloc[6,1].replace(',', '.')),0,float(info.iloc[9,1])*float(info.iloc[6,1].replace(',', '.'))] 
    print('unit conversion ='+str(units_conversion))
    print('fov ='+str(fov))
    
    # plot imported data
    plots = [None for _ in range(6)] #for colorbars
    ticks = [None for _ in range(6)] #for colorbars
    ticklabs = [['$y\'_{i\mathrm{,min}}$','$y\'_{i\mathrm{,max}}$'],['$e_{i\mathrm{,max}}$','$e_{i\mathrm{,min}}$'],['$w_{i\mathrm{,min}}$','$w_{i\mathrm{,max}}$'],['$w_{i\mathrm{,min}}$','$w_{i\mathrm{,max}}$'],['no hole','hole'],['$y\'_{i\mathrm{,min}}$','$y\'_{i\mathrm{,max}}$']]
    pltst.figure((13.72,7),'LSM_analysis') 
    fig,ax = plt.subplots(ncols=3,nrows=2)
    ax = ax.flat #ax = ax.flatten(order='F')
    ax[0].set(title='Imported data')
    #ax1.set(ylabel='y / $\mathrm{\mu m}$', xlabel='x / $\mathrm{\mu m}$')
    plots[0] = ax[0].imshow(df,extent=fov)
    ticks[0] = [df.min().min(),df.max().max()]


    # %% filter data
    size_mat = 3 # size of convolution matrix
    
    import filters as fi
    df_filt = fi.medianfilter(df, size_mat)
    
    # %% fit plane with weighted least square
    
    start_time2 = time.time()
    
    import weightedleastsquares as wlsq
    
    # choose threshold for end of itertion
    threshold_iteration = 10000
    
    # weighted least square
    X, y, y_org, beta, W, df_error_w, e, w, df_weights, df_depth = wlsq.weightedlsq(df_filt, threshold_iteration)
    
    #plots
    ax[1].set(title='WLS error')
    plots[1] = ax[1].imshow(df_error_w,extent=fov)
    ticks[1] = [df_error_w.min().min(),df_error_w.max().max()]
    #ax10.set(xlim=x_lim)
    
    run_time_wls = time.time()-start_time2
    print('Laufzeit least square:', run_time_wls, 'Sekunden')
      
    # %% filter data 2nd time

    if magnification == 10:
        size_mat = int(24/n_resize) # size of convolution matrix
        #print('Achtung: Code läuft für Bilder mit 10-facher Vergrößerung nicht optimal')
        
    elif magnification == 20:
        size_mat = int(36/n_resize)
        
        # convolutional matrix should have odd numbers
        if size_mat % 2: #ungerade
            size_mat = size_mat
        else: #gerade
            size_mat = size_mat-1
            #print('size_mat gerade: ',size_mat)

    elif magnification == 50:
        size_mat = int(120/n_resize) # size of convolution matrix
        #print('Achtung: Code läuft für Bilder mit 50-facher Vergrößerung nicht optimal')

    # size_mat = int(20/(2**(n_resize-1)))
    size_mat = 5

    #import filters as fi
    df_weights_filt = fi.medianfilter(df_weights, size_mat)
    
    #plot
    plots[2] = ax[2].imshow(df_weights,extent=fov)
    ticks[2] = [df_weights.min().min(),df_weights.max().max()]
    ax[2].set(title='WLS weights')
    plots[3] = ax[3].imshow(df_weights_filt,extent=fov)
    ticks[3] = [df_weights_filt.min().min(),df_weights_filt.max().max()]
    ax[3].set(title='Smoothed WLS weights')

   # ax11.set(xlim=x_lim)
    #plt.colorbar()
    #%% segmentation via weights
    
    # threshold for weights
    import referenceplane as ref
    # lower referenceplane
    prozent_ref_l = 0.001 # portion of datapoints to calculate reference plane 0.001
    ref_l_weights = ref.lower_ref(df_error_w, prozent_ref_l)['Z']
    print('ref_l_weights', ref_l_weights)
    
    # choose threshold for weights
    if ref_l_weights > -10:
        #threshold_weights = 0.01 0.1555
        threshold_weights = 0.1555
    else:  #everything under the threshold is classified as hole, i.e. the larger the threshold the larger the holes or the larger the threshold the more stuff is clasified as holes
        threshold_weights = 2.5e-05
        #threshold_weights = 1e-11 #testing
        #threshold_weights = 5.2536 * np.exp(0.352 * ref_l_weights) #analytical formula by Paul
        #threshold_weights = 100 * np.exp(0.352 * ref_l_weights) #adjusted analytical formula
        #threshold_weights = 1/(10**(max_val/10 - 4.3))
    
    if magnification == 50:
        threshold_weights = adaptive_thresholding(df_error_w, int(35/(n_resize)), 0.1)

        # threshold_weights = 0.45 #manuel choose threshold
    print('threshold_weights: ', threshold_weights)

    # classify every pixel weighed below threshold as hole
    # df_hole_w_2 = (df_error_w < threshold_weights and df_error_w < pd.DataFrame(np.zeros(df_error_w.shape[1], df_error_w.shape[0]))) * 1
    df_hole_w_2 = (df_error_w < threshold_weights) * 1
    
    # plot 
    ax[4].set(title='Segmentation')
    #ax2.set(ylabel='y / $\mathrm{\mu m}$', xlabel='x / $\mathrm{\mu m}$')
    #ax2.set(xlim=x_lim)
    plots[4] = ax[4].imshow(df_hole_w_2,extent=fov)
    ticks[4] = [df_hole_w_2.min().min(),df_hole_w_2.max().max()]
    
    for i in [0,1,2,3,4]:
        if i != 2:
            ax[i].annotate('', xy=(1.12,0.5), xytext=(1.03,0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=7, headlength=7))
        scalebar = AnchoredSizeBar(ax[i].transData,
                                       50, '50 $\mathrm{\mu m}$', 'lower right', 
                                       pad=0.3,
                                       color='black',
                                       frameon=True,
                                       label_top=True)
                                       #size_vertical=1)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].add_artist(scalebar)

    
    #ax[2].annotate('', xy=(1.12,0.5), xytext=(1.03,0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0))    
    #ax[3].annotate('', xy=(-0.03,0.5), xytext=(-0.11,0.5), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0))
    ax[2].annotate('', xy=(0.5,-0.06), xytext=(0.5,-0.03), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0.5))
    ax[3].annotate('', xy=(0.5,1.09), xytext=(0.5,1.15), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=7, headlength=7))
    ax[3].annotate('', xy=(0.5,1.15), xytext=(2.8,1.15), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0.5))
    #%% group holes
    
    start_time3 = time.time()
    
    import groupholes
    
    # threshold for filter shallow holes
    import referenceplane as ref
    # lower referenceplane
    prozent_ref_l = 0.1 # portion of datapoints to calculate reference plane
    ref_l_filt = ref.lower_ref(df_error_w, prozent_ref_l)['Z']
    print('ref_l_filt', ref_l_filt)
    
    #threshold for filter small holes 
    n_pixel = df.shape[0]*df.shape[1]
    threshold_smallholes = round(n_pixel/(19660.8 / n_resize))
    print('threshold_smallholes', threshold_smallholes)
    
    # choose search-matrix size
    mat_col = 2
    mat_lin = 8 #round(hole_spacing / 3)
    
    df_groups, n_holes = groupholes.groupholes(df_hole_w_2, mat_lin, mat_col, df_error_w, threshold_smallholes, ref_l_filt)
    print(n_holes)
    run_time_hole_numbering = time.time()-start_time3
    print('Laufzeit Number holes:', run_time_hole_numbering, 'Sekunden')
    
    #%% shape analysis
    
    start_time4 = time.time()
    
    import centerofgravity
    
    # choose number of deepest hole-pixels for average depth calculation
    K = 2 #2 #5
    K_pro = 0.5
    
    df_output, df_controll, list_dist, circles,  text_dists, x1_centers, x2_centers, ns = centerofgravity.center(df, df_groups, n_holes, df_error_w, K, K_pro, time)
    
    for m, n in enumerate(ns):
        ax[5].add_artist(circles[m]) # add circle to plot
        ax[5].text(x1_centers[m] + text_dists[m] , x2_centers[m] - text_dists[m] , n, color="red", fontsize=pltst.fontsizeannotate, clip_on=True)    # add hole number to plot

    ax[5].set(title='Geometries')
    # plots[5] = ax[5].imshow(df)    # show color-map as background in 6-th figure
    plots[5] = ax[5].imshow(photo, extent=[0, df.shape[1], df.shape[0], 0])    # show original LSM-foto as background

    ticks[5] = [df.min().min(),df.max().max()]
    print(plots)
    print(ticks)
    print(ticklabs)
    
    print('n resize = ' + str(n_resize))
    
    scalebar = AnchoredSizeBar(ax[5].transData,
                                   50/n_resize/units_conversion, '50 $\mathrm{\mu m}$', 'lower right', 
                                   pad=0.3,
                                   color='black',
                                   frameon=True,
                                   label_top=True)
                                   #size_vertical=1/n_resize/units_conversion)
    ax[5].set_xticklabels([])
    ax[5].set_yticklabels([])
    ax[5].add_artist(scalebar)
    for a, number in enumerate(['a','b','c','d','e','f']):
        ax[a].annotate(text=number,xy=(0.01,0.94),color='black',xycoords='axes fraction',bbox=dict(facecolor='white',pad=2))
        ax[a].annotate(text='                                 \n',xy=(0.56, 0.881),fontsize=10,color='black',xycoords='axes fraction',bbox=dict(facecolor='white',pad=2))
        #rect = patches.Rectangle((50, 100), 40, 30, edgecolor='k', facecolor='w') 0.56,0.9
        #ax[a].add_patch(rect)
        if number == 'e':
            patches = [Patch(color=pltst.cmap(0)),Patch(color=pltst.cmap(0.9999999))]
            ax[a].legend(handles=patches, labels=['no hole','hole'], loc='upper right', ncol=2, framealpha=1, fontsize=pltst.fontsizeannotate, frameon=False, handlelength=1, columnspacing=0.8, borderpad=0.3)
        else:
            axins = inset_axes(ax[a],
                                width="30%",  
                                height="2%",
                                loc='upper right',
                                bbox_to_anchor=(-0.07,-0.02,1,1), #(1.05, 1, 1, 1),
                                bbox_transform=ax[a].transAxes,
                                borderpad=0,
                               )
            cbar = plt.colorbar(mappable = plots[a], cax=axins, orientation="horizontal", ticks=ticks[a]) #, fontsize = pltst.fontsizeannotate) #pad=0.01,  fraction=0.04*im_ratio) ax=ax[:,0] #location = 'bottom' cbar = fig.colorbar(plot, ax=ax[:,0], location="bottom")
            cbar.set_ticklabels(ticklabs[a], fontsize=pltst.fontsizeannotate)
    plt.savefig(path_export+'\\'+filename+'_result.'+file_format,dpi=1200,format=file_format)
   
    #fig3,ax3 = plt.subplots()
    #plt.title('Result: ' + filename)
    #ax3.set(ylabel='y / pixel', xlabel='x / pixel')
    #plt.imshow(df)
    #plt.tight_layout()
    #plt.savefig(path_export+'\\'+filename+'_centerofgravity.svg')
    
    # delete misdetected holes from output matrix
    df_output.drop(df_output.index[df_output['Lochnummer'] == 0], inplace = True)
    df_output = df_output.reset_index(drop=True)
    
    # convert units 
    df_output['Durchmesser'] = df_output['Durchmesser'] * n_resize * units_conversion # convert diameter
    df_output['Std_Durchmesser'] = df_output['Std_Durchmesser'] * n_resize * units_conversion # convert std_diameter
    
    run_time_geometric_center = time.time()-start_time4
    print('Laufzeit Geometric center:', run_time_geometric_center, 'Sekunden')
    
    #%% calculate results
    try:
        res_depth = sum(df_output['Tiefe'])/df_output.shape[0]
        res_diameter = sum(df_output['Durchmesser'])/df_output.shape[0]
    except ZeroDivisionError: #avoid error in case no holes were identified
        res_depth = 0
        res_diameter = 0
    
    #%% calculate standard deviation
    
    std_depth = np.std(df_output['Tiefe']) 
    std_diameter = np.std(df_output['Durchmesser']) 
    
    #%% close all figures
    if show == False:
        plt.close('all')
       
    #%% end timer
    run_time_total = time.time()-start_time 
    print('Laufzeit Gesamt:', run_time_total, 'Sekunden')
    
    return res_depth, res_diameter, df_output, std_depth, std_diameter, df_error_w, df_groups, df_weights, info, df, df_weights_filt, df_hole_w_2, df_controll, list_dist, y, data_size, run_time_wls, run_time_hole_numbering, run_time_geometric_center, run_time_total


def adaptive_thresholding(df_to_process, mat_size, C):
    from scipy import ndimage
    # Thresholding parameters

    threshold_output = ndimage.median_filter(df_to_process, mat_size) - C

    return threshold_output