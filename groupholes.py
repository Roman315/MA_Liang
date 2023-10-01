import pandas as pd
import numpy as np

def groupholes(df_hole, mat_lin, mat_col, df_error_w, threshold_smallholes, ref_l_filt):
    
    columns = df_hole.shape[1]
    lines = df_hole.shape[0]
    
    
    df_groups = pd.DataFrame(0, index = (range(lines + 2*mat_lin)), columns = (range(columns + 2*mat_col))) # create 0-Dataframe with added rows and columns, so search matrix doesn't search outside of frame
    i = 0
    n = 0
    neighborcell = 'NO'
    
    while i < columns: # loop for columns
        j = 0
        while j < lines: # loop for lines
            
            if df_hole.iloc[j][i] == 1 and df_error_w.iloc[j][i] < 0: # if hole was found
                
                # check neighborcells for hole
                l = round(-(mat_lin/2))+1 # Hinterfragen: Unterschied gerade ungerade Zahl
                while l <= mat_lin/2: # loop for lines of search-matrix
                    k = 0
                    while k < mat_col: # loop for columns of search-matrix
                        if df_groups.iloc[j + mat_lin + l, i + mat_col - k] > 0: # if: hole-pixel was detected in one of the neighborcells
                            m = df_groups.at[j + mat_lin + l, i + mat_col - k] # m is the number of the detected neighbor-hole-pixel
                            neighborcell = 'YES'
                            
                            k = mat_col #end loop k
                            l = mat_lin #end loop l
                        else: # else: continue search
                            neighborcell = 'NO'
                            k = k + 1
                    l = l+1
                
                # if neighborcell is a hole, copy holenumber of neighborpixel
                if neighborcell == 'YES':
                    while df_hole.iloc[j, i] == 1:
                        df_groups.at[j + mat_lin, i + mat_col] = m
                        
                        if j < lines-1: # nicht so schöne Problembehebung
                            j = j+1
                        else:
                            break
                        
                # else new hole
                else:
                    n = n + 1 # new hole number
                    m = n # update m
                    #df_groups.at[j,i] = m
                    while df_hole.iloc[j, i] == 1:
                        df_groups.at[j + mat_lin, i + mat_col] = m
                        
                        if j < lines-1: # nicht so schöne Problembehebung
                            j = j+1
                        else:
                            break
                        
            j = j+1
        i = i+1
     
    # delete added columns/rows
    df_groups = df_groups.drop(df_groups.tail(mat_lin).index) # delete last mat_lin-rows
    df_groups = df_groups.drop(df_groups.head(mat_lin).index) # delete first mat_lin-rows
    df_groups = (df_groups.iloc[:,:-mat_col]) # delete last mat_col-columns
    df_groups = (df_groups.iloc[:,mat_col:]) # delete first mat_col-columns
    
    df_groups = df_groups.reset_index(drop=True) # reset row-names starting at 0
    df_groups.columns = [np.arange(0,df_groups.shape[1])] #reset column-names starting at 0

    '''# filter small and shallow holes
    for i in range(1, n+1):
        if (df_groups.isin([i])).sum().sum() < threshold_smallholes: # if hole has very few pixel

            # calculate maximal depth of hole
            arr_err = df_error_w.to_numpy()
            arr_gr = df_groups.to_numpy()
            max_depth = np.amin(arr_err[arr_gr == i])
            
            if max_depth > ref_l_filt: # if hole is very shallow
                df_groups = df_groups.replace(i, 0)'''
    
    # filter holes at image border
    arr_gr = df_groups.to_numpy()
    arr_border = np.concatenate([arr_gr[0], arr_gr[lines-1], arr_gr[:,0], arr_gr[:,columns-1]])
    arr_uni = np.unique(arr_border)
    for i in range(1, len(arr_uni)):
        df_groups = df_groups.replace(arr_uni[i], 0)
    
    
    # renumber filtered holes
    j=1
    for i in range(1, n+1):
        if (df_groups == i).sum().sum() > 0: 
            df_groups = df_groups.replace(i, j)
            j = j+1       
    n = j-1
    
    
    return df_groups, n