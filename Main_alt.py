def main(filename, path_to_file, path_export, hole_spacing, n_resize, show, file_format):
    
    #%% start timer
    import time
    start_time = time.time()
    
    # %% import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import sys #for organizing paths
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sys.path.append(r'H:\05_Forschung\03_Programme\01_Auswertungen\00_Plotting')
    import plotstyle as pltst #import custom plotstyle
    #x_lim=(100,650)
    #y_lim=()
    
    # %% import data
    import importdata as im
    
    df, info, magnification, units_conversion, hole_spacing, data_size = im.import_data(path_to_file, n_resize, hole_spacing) # import data as pandas dataframes
    
    #determine field of view from imported data
    fov = [0,float(info.iloc[8,1])*float(info.iloc[6,1].replace(',', '.')),0,float(info.iloc[9,1])*float(info.iloc[6,1].replace(',', '.'))] 
    print('unit conversion ='+str(units_conversion))
    print('fov ='+str(fov))
    # plot imported data
    pltst.figure((13.72,7),'LSM_analysis') 
    fig,ax = plt.subplots(ncols=3,nrows=2)
    ax = ax.flat #ax = ax.flatten(order='F')
    ax[0].set(title='Imported data')
    #ax1.set(ylabel='y / $\mathrm{\mu m}$', xlabel='x / $\mathrm{\mu m}$')
    ax[0].imshow(df,extent=fov)
    #ax1.set(xlim=x_lim)

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
    X, y, y_org, beta, W, df_error_w, e, w, df_weights = wlsq.weightedlsq(df_filt, threshold_iteration)
    
    #plots
    ax[1].set(title='WLS error')
    ax[1].imshow(df_error_w,extent=fov)
    #ax10.set(xlim=x_lim)
    
    run_time_wls = time.time()-start_time2
    print('Laufzeit least square:', run_time_wls, 'Sekunden')
      
    # %% filter data 2nd time

    if magnification == 10:
        size_mat = int(24/n_resize) # size of convolution matrix
        print('Achtung: Code läuft für Bilder mit 10 facher Vergrößerung nicht optimal')
        
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
        print('Achtung: Code läuft für Bilder mit 50 facher Vergrößerung nicht')
    
    size_mat = 5
    
    #import filters as fi
    df_weights_filt = fi.medianfilter(df_weights, size_mat)
    
    #plot
    ax[2].imshow(df_weights,extent=fov)
    ax[2].set(title='WLS weights')
    ax[3].imshow(df_weights_filt,extent=fov)
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
        threshold_weights = 2.5e-05 #Example AnaGeo
        #threshold_weights = 1e-11 #testing
        #threshold_weights = 5.2536 * np.exp(0.352 * ref_l_weights) #analytical formula by Paul
        #threshold_weights = 100 * np.exp(0.352 * ref_l_weights) #adjusted analytical formula
        #threshold_weights = 1/(10**(max_val/10 - 4.3))
    
    if magnification == 50:
        threshold_weights =  0.45 #manuel choose threshold best my data 0.4 oder 0.5
    
    print('threshold_weights: ', threshold_weights)
    
    # classify every pixel weighed below threshold as hole
    df_hole_w_2 = (df_weights_filt < threshold_weights) * 1
    
    # plot 
    ax[4].set(title='Segmentation')
    #ax2.set(ylabel='y / $\mathrm{\mu m}$', xlabel='x / $\mathrm{\mu m}$')
    #ax2.set(xlim=x_lim)
    ax[4].imshow(df_hole_w_2,extent=fov)
    
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
    mat_lin = 12 #round(hole_spacing / 3)
    
    df_groups, n_holes = groupholes.groupholes(df_hole_w_2, mat_lin, mat_col, df_error_w, threshold_smallholes, ref_l_filt)
    
    run_time_hole_numbering = time.time()-start_time3
    print('Laufzeit Number holes:', run_time_hole_numbering, 'Sekunden')
    
    #%% shape analysis
    
    start_time4 = time.time()
    
    import centerofgravity
    
    # choose number of deepest hole-pixels for average depth calculation
    K = 2 #2 #5
    K_pro = 0.5
    
    df_output, df_controll, list_dist, circles, text_dists, x1_centers, x2_centers, ns = centerofgravity.center(df, df_groups, n_holes, df_error_w, K, K_pro, time)
    
    for m, n in enumerate(ns):
        ax[5].add_artist(circles[m]) # add circle to plot
        #ax[5].text(x1_centers[m] + text_dists[m] , x2_centers[m] - text_dists[m] , n, color="red", fontsize=pltst.fontsizeannotate, clip_on=True) # add hole number to plot
    
    ax[5].set(title='Geometries')
    ax[5].imshow(df)
    print('n resize = ' + str(n_resize))
    scalebar = AnchoredSizeBar(ax[5].transData,
                                   50/n_resize/units_conversion, '50 $\mathrm{\mu m}$', 'lower right', 
                                   pad=0.3,
                                   color='black',
                                   frameon=True,
                                   label_top=True)
                                   #size_vertical=0.5/n_resize/units_conversion)
    ax[5].set_xticklabels([])
    ax[5].set_yticklabels([])
    ax[5].add_artist(scalebar)
    for a, number in enumerate(['a','b','c','d','e','f']):
        ax[a].annotate(text=number,xy=(0.01,0.94),color='black',xycoords='axes fraction',bbox=dict(facecolor='white',pad=2))
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
    
        
    fig1,ax1 = plt.subplots(figsize=(9,5))
    plot = ax1.imshow(df,extent=fov)
    scalebar = AnchoredSizeBar(ax1.transData,
                                   50, '50 $\mathrm{\mu m}$', 'lower right', 
                                   pad=0.3,
                                   color='black',
                                   frameon=True,
                                   label_top=True)
                                   #size_vertical=0.5/n_resize/units_conversion)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.add_artist(scalebar)
    axins = inset_axes(ax1,
                        width="5%",  
                        height="100%",
                        loc='center right',
                        borderpad=-3
                       )
    cbar = fig1.colorbar(plot, cax=axins, orientation="vertical", shrink = 0.9) #pad=0.01,  fraction=0.04*im_ratio) ax=ax[:,0] #location = 'bottom' cbar = fig.colorbar(plot, ax=ax[:,0], location="bottom")
    cbar.set_label(r'$\mathrm{Height}$ / $\mathrm{\mu m}$')
    plt.savefig(r'H:\05_Forschung\00_Messdaten\99_Sonstiges\my\Results\Doksem_my.'+file_format,dpi=1200,format=file_format)
    
    return res_depth, res_diameter, df_output, std_depth, std_diameter, df_error_w, df_groups, df_weights, info, df, df_weights_filt, df_hole_w_2, df_controll, list_dist, y, data_size, run_time_wls, run_time_hole_numbering, run_time_geometric_center, run_time_total