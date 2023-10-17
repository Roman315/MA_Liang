import matplotlib.pyplot as plt
import numpy as np
import sys #for organizing paths
sys.path.append(r'H:\05_Forschung\03_Programme\01_Auswertungen\Plotting')
import plotstyle as pltst #import custom plotstyle
colors = ['#3070b3','#A2AD00','#E37222','#DAD7CB']
    
def barplot(path_export,list_final,file_format):
    
    x = np.arange(len(list_final)) 
    
    if len(x) > 5:
        pltst.figure('doublecolumn','bar')
    else:
        pltst.figure('single','bar')
    
    # create np.array for plot
    list_res = []
    list_name = []
    list_std = []
    for count in range(len(list_final)):
        list_res.append([list_final[count][1], list_final[count][2]])
        list_std.append([list_final[count][4], list_final[count][5]])
        list_name.append(list_final[count][0]) #list_name.append([list_final[count][0]])
    arr_res = np.array(list_res)
    str_name = (list_name)
    arr_std = np.array(list_std)
    
    #plot
    fig6,ax6 = plt.subplots()
    ax6.bar(x - pltst.barwidth/4, arr_res[:, 0], width = pltst.barwidth/2, label='Tiefe', yerr = arr_std[:, 0], color=colors[0]) 
    ax6.bar(x + pltst.barwidth/4, arr_res[:, 1], width = pltst.barwidth/2, label='Durchmesser', yerr = arr_std[:, 1], color=colors[1])

    if len(x) > 5:
        plt.xticks(x, str_name, rotation ='vertical')
    else:
        plt.xticks(x, str_name)
    ax6.set_ylabel('Größe / $\mathrm{\mu m}$')
    ax6.legend()
    plt.tight_layout()
    plt.savefig(path_export+'/Results.'+file_format,dpi=1200,format=file_format)
    plt.show()

    # calculate average diameter and depth
    average_diameter = sum(arr_res[:, 1])/len(arr_res[:, 1])
    average_depth = sum(arr_res[:, 0])/len(arr_res[:, 0])

    print('Durschnittlicher Durchmesser: ', average_diameter)
    print('Durschnittliche Tiefe: ', average_depth)