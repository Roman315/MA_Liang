import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def center(df, df_groups, n_holes, df_error_w, K, K_pro, time):
    
    # create output dataframe
    df_output = np.zeros((n_holes + 1, 5))
    df_output = pd.DataFrame(df_output)
    df_output = df_output.rename(columns={0: "Lochnummer", 1: "Durchmesser", 2: "Tiefe", 3: "Std_Durchmesser", 4: "Std_Tiefe"})
    
    # create dataframe for controll edge search
    columns = df_groups.shape[1] # number of columns from dataframe
    lines = df_groups.shape[0] # number of lines from dataframe
    df_controll = np.zeros((lines, columns))
    df_controll = pd.DataFrame(df_controll)
    
    # prepare plot
    #fig5,ax5 = plt.subplots()
    
    #create empty list outside of loop to avoid assignment error in case of n = 0
    list_dist = []
    circles = []
    text_dists = []
    x1_centers = []
    x2_centers = []
    ns = []

    # loop to regard center of gravity hole by hole
    n = 1 # start hole
    while n <= n_holes: # loop for holes

        # create temporary dataframe for center of gravity
        df_center = pd.DataFrame(0, index=range(lines), columns=range(columns))
        
        # search df_groups for current holenumber (going through every column from top to bottom)
        #c = 0 # startcolumn
        #while c < columns:
        #    l=0 # startline
        #    while l < lines:
        #        if df_groups.iloc[l,c]==n: # if holenumber found
        #            df_center.at[l,c] = 1 # write 1 in df_center
        #        l = l+1
        #    c = c+1
        
        df_center = (df_groups == n) * 1
        
        '''
        # OLD CALCULATION of center of gravity
        start_time10 = time.time()
        # prepare data (in matrices/vectors) for calcultion of center of gravity
        df_center_long = df_center.unstack().reset_index() # e.g. from 768x1024 to 786432x3 Matrix
        df_center_long = df_center_long.to_numpy() # pandas to numpy
        x1 = df_center_long[:,0] 
        x2 = df_center_long[:,1] 
        y = df_center_long[:,2]     
        X = np.vstack([x1, x2]).T # ?x2 Matrix containing all pixel-positions
        X = np.matrix(X) 
        y = np.matrix(y).T # ?x1 Matrix containing 0 or 1 (1 means current hole was found in previous search loop)
        
        # calculate center of gravity
        center = X.T*y/sum(y)
        
        # "coordinates" of center of gravity for current hole
        x1_center = float(center[0])
        x2_center = float(center[1])
    
        print('Laufzeit centercoordinates-calculationOLD:',n , '.Loch:', time.time()-start_time10, 'Sekunden')
        '''
        
        #'''
        # NEW CALCULATION WITH SCIPY-LIBRARY        
        # calculate center of gravity
        df_center_np = df_center.to_numpy()
        center = ndimage.measurements.center_of_mass(df_center_np)
        
        # "coordinates" of center of gravity for current hole
        x1_center = float(center[1])
        x2_center = float(center[0])

        #print('Laufzeit centercoordinates-calculationNEW:',n , '.Loch:', time.time()-start_time12, 'Sekunden')
        #print('Abweichung:', x1_centerOLD-x1_center, x2_centerOLD-x2_center)
        #'''
        
        
        # create temporary dataframe for depth
        df_depth = np.zeros((lines, columns))
        df_depth = pd.DataFrame(df_depth)
        
        # loop to calculate distance of hole-pixel at the border to center of gravity (going through every column from top to bottom)
        start_time11 = time.time()
        #dist_sum = 0 # set sum of distances to 0 for every new hole
        #count_dist = 0
        c = 0 # startcolumn
        list_dist = []
        while c < columns:
            l = 0 # startline
            while l < lines:
                if df_center.iloc[l,c]==1: # holepixel found
                    
                    # start searching in Moore neighborhood for 0 (=>edgepixel)
                    c_search = c - 1
                    while c_search < c + 2:
                        
                        l_search = l - 1
                        while l_search < l + 2:
                            
                            if df_center.iloc[l_search,c_search]==0: #edgepixel found in Moore neighborhood
                                
                                df_controll.at[l,c] = 1 # mark found edgepixel
                                dist = np.sqrt(((c-x1_center))**2 + ((l-x2_center))**2) # distance of each point calculated with pythagoras
                                
                                #dist_sum = dist_sum + dist # sum up distances every iteration
                                #count_dist = count_dist + 1
                                list_dist.append(dist)
                                
                                #end search
                                l_search = l + 2
                                c_search = c + 2
                            
                            l_search = l_search + 1
                            
                        c_search = c_search + 1
                    
                    
                    
                    # save depth
                    depth = df_error_w.at[l,c] # extract depth from df_error
                    df_depth.at[l,c] = depth # save depth in dataframe
                l = l+1
            c = c+1
        
        #print('Laufzeit distance-calculation loop:',n , '.Loch:', time.time()-start_time11, 'Sekunden')
        
        # calculate radius
        radius = np.mean(list_dist)
        std_radius = np.std(list_dist)
        
        circle = plt.Circle((x1_center, x2_center), radius, color='r', fill=False)    # show circle specific number
        
        # prepare average depth calculation
        df_depth_long = df_depth.unstack().reset_index() # e.g. from 768x1024 to 786432x3 Matrix
        y_depth_max = df_depth_long.nsmallest(K, 0) # sorts K deepest pixels in descending order
        N = y_depth_max[y_depth_max[0] < 0].shape[0] # N = Number of pixels smaller than 0 (in case hole consistst of less pixels than K)
        
        
        # calculate average depth
        aver_depth = np.mean(y_depth_max[0])
        std_depth_hole = np.std(y_depth_max[0])
        
        #print(aver_depth)
        
        # Falls Tiefe über prozentualen Lochanteil bestimmt werden soll
        '''
        N_hole = df_depth_long[df_depth_long[0] < 0].shape[0]
        K_2 = round(N_hole*K_pro) #IST Runden in Ordnung?
        y_depth_max_pro = df_depth_long.nsmallest(K_2, 0) # sorts K deepest pixels in descending order
        N_pro = y_depth_max_pro[y_depth_max_pro[0] < 0].shape[0] # N = Number of pixels smaller than 0 (in case hole consistst of less pixels than K)
        if N_pro < 1: # to avoid devision by zero
            N_pro = 1
            
        # calculate average depth über prozent
        aver_depth_pro = sum(y_depth_max_pro[0])/N_pro #Problem bei N_pro=0
        '''
        
        # save holenumber, radius, depth in output dataframe
        df_output.at[n, 'Lochnummer'] = n
        df_output.at[n, 'Durchmesser'] = radius * 2
        df_output.at[n, 'Tiefe'] = aver_depth
        df_output.at[n, 'Std_Durchmesser'] = std_radius*2 
        df_output.at[n, 'Std_Tiefe'] = std_depth_hole
        #df_output.at[n, 'Tiefe2'] = aver_depth_pro
        
        # prepare plot
        #ax5.add_artist(circle) # add circle to plot
        text_dist = np.sqrt(2)/2*radius # distance for position of number in plot
        #ax5.text(x1_center + text_dist , x2_center - text_dist , n, color="red") # add hole number to plot
        circles.append(circle)
        text_dists.append(text_dist)
        x1_centers.append(x1_center)
        x2_centers.append(x2_center)
        ns.append(n)
        
        n = n + 1 # increase holenumber for next iteration
    
    return df_output, df_controll, list_dist, circles,  text_dists, x1_centers, x2_centers, ns
