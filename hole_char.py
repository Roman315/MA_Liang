'''
    Characterization of the detected holes:
        1. Find holes
        2. Find corresponding weights of the holes
        3. Calculate diameters
        4. Adaptive thresholding
'''

# Packages to be used
from scipy import ndimage
import math
import numpy as np

def find_holes(df_to_process, filter_window_hole_search):

    threshold = adaptive_thresholding(df_to_process)
    imprints = (ndimage.median_filter(df_to_process, filter_window_hole_search)[:,:] < threshold)

    df_groups = np.zeros_like(df_to_process)
    hole_number = -1
    hole_number_nearby = 0
    x_boundary = [[]]
    y_boundary = [[]]
    x_all_points = [[]]
    y_all_points = [[]]
    x_size = df_to_process.shape[0]
    y_size = df_to_process.shape[1]
    k = 0
    while k < imprints.shape[1]:
        i = 0
        while i < imprints.shape[0]:
            hole_is_found_nearby = False    #reset the flag, whether a hole is found nearby
            if imprints[i, k] == True:    #hole is detected
                
                for i_1 in range(1, 3, 1):
                    if (0 <= (i-i_1) < df_groups.shape[0]) and (imprints[i-i_1, k] == True):
                        hole_number_nearby = df_groups[i-i_1, k]
                        hole_number_nearby = int(hole_number_nearby)
                        hole_is_found_nearby = True    #mark whether hole is found in nearby area
                        break
                for i_1 in range(-3,3):    #detection in nearby area
                    for k_1 in range(1,3,1):
                        if ((0 <= (i-i_1) < df_groups.shape[0]) and (0 <= (k-k_1) < df_groups.shape[1])) and (imprints[i-i_1, k-k_1] == True):
                            hole_number_nearby = df_groups[i-i_1, k-k_1]
                            hole_number_nearby = int(hole_number_nearby)
                            hole_is_found_nearby = True    #mark whether hole is found in nearby area
                            break
                    else:
                        continue
                    break

                if hole_is_found_nearby == True:
                    df_groups[i, k] = hole_number_nearby
                    
                    # Arrays to save the index of the first point occoured in a coloumn, which is also boundary indexes
                    x_boundary[hole_number_nearby].append(i)
                    y_boundary[hole_number_nearby].append(k)
                    
                    # Arrays to save the indexes of the first point occoured in a coloumn
                    x_all_points[hole_number_nearby].append(i)
                    y_all_points[hole_number_nearby].append(k)
                    
                    # The pixel immediately below, which is also a hole, is also marked with the same number.
                    i += 1    #the next point
                    for j in range(1, imprints.shape[0] - i):
                        if imprints[i, k] == True:
                            df_groups[i, k] = hole_number_nearby
                            x_all_points[hole_number_nearby].append(i)
                            y_all_points[hole_number_nearby].append(k)
                            i += 1

                        else:
                            # The last point occoured in the coloumn with the specific hole number, which is boundary.
                            x_boundary[hole_number_nearby].append(i - 1)
                            y_boundary[hole_number_nearby].append(k)
                            break
                else:    #new hole is found
                    hole_number += 1
                    hole_number = int(hole_number)
                    df_groups[i, k] = hole_number
                    
                    if hole_number > 0:
                        x_boundary.append([])
                        y_boundary.append([])
                        x_all_points.append([])
                        y_all_points.append([])
                    
                    x_all_points[hole_number].append(i)
                    y_all_points[hole_number].append(k)
                    x_boundary[hole_number].append(i)
                    y_boundary[hole_number].append(k)
                    
                    # The pixel immediately below, which is also a hole, is also marked with the same number.
                    i += 1
                    for j in range(1, imprints.shape[0]-i): 
                        if imprints[i, k] == True:
                            df_groups[i, k] = hole_number
                            x_all_points[hole_number].append(i)
                            y_all_points[hole_number].append(k)
                            i += 1

                        else:
                            x_boundary[hole_number].append(i - 1)
                            y_boundary[hole_number].append(k)
                            break
            else:
                i += 1      
        k += 1
        
    # Classify konvex shape of the countour to one holes:
    # For konvex shape, the above classify-method may devide konvex shape to two groups of holes, which is not accurate
    # for the detection of diameters and depth. The under algorithum will make the seperated konvex shape to one hole.
    detected_num_holes = len(x_all_points)
    num_holes = int(0)
    while num_holes < detected_num_holes:
        
        i = -1
        while i < (len(x_all_points[num_holes]) - 1):    #Every points in the hole with number num_holes
            i += 1
            
            for row in range(-1,2):
                for column in range(-1,2):
                    #if a point is caught in the middle
                    if (0 <= (x_all_points[num_holes][i] + row) < df_groups.shape[0]) \
                    and (0 <= (y_all_points[num_holes][i] + column) < df_groups.shape[1]) \
                    and (df_groups[x_all_points[num_holes][i] + row, y_all_points[num_holes][i] + column] > num_holes):
                        num_holes_another = df_groups[x_all_points[num_holes][i] + row, y_all_points[num_holes][i] + column]
                        num_holes_another = int(num_holes_another)
                        df_groups[df_groups[:,:]==num_holes_another] = num_holes
                        #add the connected hole_another to the original hole
                        if num_holes_another >= len(x_all_points):
                            break
                        for j in range(0, len(x_all_points[num_holes_another])):
                            x_all_points[num_holes].append(x_all_points[num_holes_another][j])
                            y_all_points[num_holes].append(y_all_points[num_holes_another][j])
                        for j in range(0, len(x_boundary[num_holes_another])):
                            x_boundary[num_holes].append(x_boundary[num_holes_another][j])
                            y_boundary[num_holes].append(y_boundary[num_holes_another][j])
                        #delete the data from hole_another
                        x_all_points.pop(num_holes_another)
                        y_all_points.pop(num_holes_another)
                        x_boundary.pop(num_holes_another)
                        y_boundary.pop(num_holes_another)
                        df_groups[df_groups[:,:]>num_holes_another] -= 1
                        
        #delete the hole which contacts boundary
        if (True in [x == 0 for x in x_all_points[num_holes]]) \
        or (True in [y == 0 for y in y_all_points[num_holes]]) \
        or (True in [x == (x_size-1) for x in x_all_points[num_holes]]) \
        or (True in [y == (y_size-1) for y in y_all_points[num_holes]]):
            x_all_points.pop(num_holes)
            y_all_points.pop(num_holes)
            x_boundary.pop(num_holes)
            y_boundary.pop(num_holes)
            df_groups[df_groups[:,:]==num_holes] = 0
            df_groups[df_groups[:,:]>num_holes] -= 1
        else:
            num_holes += 1

        detected_num_holes = len(x_all_points)

    #Calculate the position(weight), diameter, and depth for each hole group
    x_weight, y_weight, radius = get_diameter(x_all_points, y_all_points, x_boundary, y_boundary, df_groups)
    # depth = get_depth(df_to_process, x_all_points, y_all_points)
        
    return df_groups, x_weight, y_weight, radius

def get_diameter(x_all_points, y_all_points, x_boundary, y_boundary, df_groups):
    x_weight = []
    y_weight = []
    R = []
    
    j = 0
    while j < len(x_all_points):
        
#     for j in range(0, len(x_all_points)):    #process each hole individually
        #the weight point location
        A = len(x_all_points[j])    #total pixes of a hole number
        x_weight.append(sum(x_all_points[j]) / A)
        y_weight.append(sum(y_all_points[j]) / A)
        dx = [x - x_weight[j] for x in x_boundary[j]]
        dx = [x**2 for x in dx]
        dy = [y - y_weight[j] for y in y_boundary[j]]
        dy = [y**2 for y in dy]
#         R.append((sum([math.sqrt(x2+y2) for x2, y2 in zip(dx, dy)])) / len(x_boundary[j]))
        
        #delete holes radius larger than 7 um
        Radius = (sum([math.sqrt(x2+y2) for x2, y2 in zip(dx, dy)])) / len(x_boundary[j])
        if Radius < (1/0.27) :    #3.5um divide magnification of 0.27 is 13
            x_all_points.pop(j)
            y_all_points.pop(j)
            x_boundary.pop(j)
            y_boundary.pop(j)
            x_weight.pop(j)
            y_weight.pop(j)
            df_groups[df_groups[:,:] == j] = 0
            df_groups[df_groups[:,:]>j] -= 1
        else:
            R.append(Radius)
            j += 1
    return x_weight, y_weight, R

def get_depth(df_to_process, x_all_points, y_all_points):
    depth = []
    for j in range(0, len(x_all_points)):
        height = []
        print(len(x_all_points))
        print(len(y_all_points))
        for j_1 in range(0, len(x_all_points[j])):
            # print(df_to_process)
            # print('x', x_all_points)
            # print(y_all_points)
            height.append(df_to_process[x_all_points[j][j_1], y_all_points[j][j_1]])

        height.sort()
        depth.append(sum(height[0:5]) / len(height[0:5]))
    return depth

def adaptive_thresholding(df_to_process):
    # Thresholding parameters
    size = 25
    C = 0.03
    
    threshold_output = ndimage.median_filter(df_to_process, size) - C
    
    return threshold_output

