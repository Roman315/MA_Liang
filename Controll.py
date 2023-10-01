import os
import matplotlib.image as image

# path_import = r'V:\Scratch\hi\my\Datenauswahl'
path_import = r'D:\MA_Liang\LSM\Proben'

#runtime settings 
n_resize = 5 #specfify resolution (ideally: n_resize = 1), use only every n-th row and column to reduce runtime, typical value for good results at a reasonable runtime: 5
path_export = path_import + r'\Results1_n_' + str(n_resize)
try:
    os.mkdir(path_export) 
    print("Directory " , path_export,  " created ") 
except FileExistsError:
    print("Directory " , path_export,  " already exists")

#figure settings
show = True #specify if figures should be shown, disable for large datsets to save RAM
file_format = 'pdf' #select in which file format the files should be saved, e.g. 'pdf', 'svg', 'png', 'jpg'

#%% start to analyse all CSV-files in given import-directory
import Main

folder_name = path_import.split('/')[len(path_import.split('/'))-2]

list_final = []

# for loop to call file after file
for root, dirs, files in os.walk(path_import):
   for name in files:
        if os.path.splitext(name)[-1] == '.csv' and 'Results' not in name:
            
            path_to_file = os.path.join(root, name)
            filename = os.path.join(name).split('.csv')[0]
            print('Dateiname: ' + filename)
            photo_path = path_to_file.replace('_Height.csv', '.png')
            print('photo path is:', photo_path)
            photo = image.imread(photo_path)    # resize photo as the same as roh data-frame: easier for figure.f to plot photo as background
            
            #start main program
            res_depth, res_diameter, df_output, std_depth, std_diameter, df_error_w, df_groups, df_weights, info, df, df_weights_filt, df_hole_w_2, df_controll, list_dist, y, data_size, run_time_wls, run_time_hole_numbering, run_time_geometric_center, run_time_total = Main.main(filename, path_to_file, path_export, n_resize, show, file_format, photo)
            
            #save results in list
            list_final.append([filename, res_depth, res_diameter, df_output, std_depth, std_diameter, data_size, run_time_wls, run_time_hole_numbering, run_time_geometric_center, run_time_total])
            
            print('Tiefe: ', res_depth)
            print('Durchmesser: ', res_diameter)
            print('')
#%% Barplot
import barplot
barplot.barplot(path_export,list_final,file_format)

#%% export
import export
export.export(path_export, list_final, n_resize)
#export.exportlumentum(path_export, list_final)
