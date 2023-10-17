def main(filename, path_to_file, path_export, n_resize, show, file_format, photo, roh_data_filter_mat_size, hole_search_filter_mat_size, show_num_with_indentations):
    
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
    import openpyxl
    
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
    roh_data_filter_mat_size = 3 # size of convolution matrix
    
    import filters as fi
    df_filt = fi.medianfilter(df, roh_data_filter_mat_size)
    
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
    
    run_time_wls = time.time()-start_time2
    print('Laufzeit least square:', run_time_wls, 'Sekunden')
      
    # %% filter data 2nd time
    filter_twice_size_mat = 5
    #import filters as fi
    df_weights_filt = fi.medianfilter(df_weights, filter_twice_size_mat)
    #plot
    plots[2] = ax[2].imshow(df_weights,extent=fov)
    ticks[2] = [df_weights.min().min(),df_weights.max().max()]
    ax[2].set(title='WLS weights')
    plots[3] = ax[3].imshow(df_weights_filt,extent=fov)
    ticks[3] = [df_weights_filt.min().min(),df_weights_filt.max().max()]
    ax[3].set(title='Smoothed WLS weights')

    #%% segmentation via weights
    # threshold for weights
    import referenceplane as ref
    # lower referenceplane
    prozent_ref_l = 0.001 # portion of datapoints to calculate reference plane 0.001
    ref_l_weights = ref.lower_ref(df_error_w, prozent_ref_l)['Z']
    print('ref_l_weights', ref_l_weights)
    
    adaptive_threshold = adaptive_thresholding(df_error_w, int(35/(n_resize)), 0.2)
    # threshold_weights = 0.45 #manuel choose threshold
    print('threshold_weights: ', adaptive_threshold)    #show adaptive threshold

    # classify every pixel weighed below threshold as hole
    # df_hole_w_2 = (df_error_w < threshold_weights and df_error_w < pd.DataFrame(np.zeros(df_error_w.shape[1], df_error_w.shape[0]))) * 1
    df_hole_w_2 = (df_error_w < adaptive_threshold) * 1    #variable used for showing the segmentation results
    
    # plot 
    ax[4].set(title='Segmentation')
    plots[4] = ax[4].imshow(df_hole_w_2, extent=fov)
    ticks[4] = [df_hole_w_2.min().min(), df_hole_w_2.max().max()]
    
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

    ax[2].annotate('', xy=(0.5,-0.06), xytext=(0.5,-0.03), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0.5))
    ax[3].annotate('', xy=(0.5,1.09), xytext=(0.5,1.15), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=7, headlength=7))
    ax[3].annotate('', xy=(0.5,1.15), xytext=(2.8,1.15), xycoords='axes fraction', horizontalalignment='center', verticalalignment='center', annotation_clip=False, arrowprops=dict(facecolor='black', width=0.5, headwidth=0.5, headlength=0.5))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Group holes and shape analysis
    start_time3 = time.time()
    import referenceplane as ref
    # threshold for filter shallow holes
    run_time_hole_numbering = time.time()-start_time3
    print('Laufzeit Number holes:', run_time_hole_numbering, 'Sekunden')
    start_time4 = time.time()
    
    import centerofgravity
    
    # choose number of deepest hole-pixels for average depth calculation
    
    df_output, df_holes_num, circles, text_dists, x1_centers, x2_centers, ns = centerofgravity.center(df_error_w, time, adaptive_threshold, hole_search_filter_mat_size)
    df_output.to_excel(path_export+'\\'+filename+'_result.xlsx')
    ns = np.arange(0, len(circles)-1)
    for m, n in enumerate(ns):
        ax[5].add_artist(circles[m]) # add circle to plot
        text_x2 = x2_centers[m] - text_dists[m]
        text_x1 = x1_centers[m] + text_dists[m]
        if show_num_with_indentations == True:
            # ax[5].text(text_x2, text_x1, m, color="red", fontsize=pltst.fontsizeannotate, clip_on=True)    # add hole number to plot
            ax[5].text(x2_centers[m], x1_centers[m], m, color="red", fontsize=5, clip_on=True)    # add hole number to plot

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
    if show_num_with_indentations == True:
        plt.savefig(path_export+'\\'+filename+'_result_with_num.'+file_format,dpi=1200,  format=file_format)
    else:
        plt.savefig(path_export+'\\'+filename+'_result.'+file_format,dpi=1200,  format=file_format)
   
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
    
    return res_depth, res_diameter, df_output, std_depth, std_diameter, df_error_w, df_weights, info, df, df_weights_filt, df_hole_w_2, df_holes_num, y, data_size, run_time_wls, run_time_hole_numbering, run_time_geometric_center, run_time_total


def adaptive_thresholding(df_to_process, mat_size, C):
    from scipy import ndimage
    # Thresholding parameters

    threshold_output = ndimage.median_filter(df_to_process, mat_size) - C

    return threshold_output