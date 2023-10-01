import matplotlib.pyplot as plt

#plot styleguide
barwidth = 0.6 #define width for bars
barlinewidth = 1.5
fontsizeannotate = 10
plt.rc('figure', dpi = 100, figsize = (6.5,4.8), autolayout = False)
plt.rc('figure.constrained_layout', use = False, h_pad = 0.1, hspace = 0.1, w_pad = 0.005, wspace = 0.005)
plt.rc('figure.subplot', top = 0.9, bottom = 0.1, left = 0.2, right = 0.8, hspace = 0.2, wspace = 0.2) #left = 0.12, right = 0.88 hspace = 0.15 for ProzInt for AnaGeo: top = 0.85, bottom = 0.1, left = 0.2, right = 0.8, hspace = 0.2, wspace = 0.25
plt.rc('font', size = 11, family = 'sans-serif') #choose between "serif" and "sans-serif"
plt.rc('xtick', top = True, direction = 'in', labelbottom = True)
plt.rc('ytick', right = True, direction = 'in', labelleft = True)
plt.rc('lines', linewidth = 1.1, markersize = 5.0, marker = 'o')
plt.rc('axes', titlesize = 'medium', grid = True, axisbelow=True)
plt.rc('image', cmap='viridis')
plt.rc('errorbar', capsize = 3)
#plt.rc('legend', handletextpad = 0.3)
plt.rc('legend', fancybox = False, framealpha = None, edgecolor = 'k', handletextpad = 0.05, columnspacing = 0.1)#, fontsize=fontsizeannotate) #change legend style to classic layout  handletextpad = 0.05, columnspacing = 0.1)

#plt.rc('patch', force_edgecolor = False, linewidth = 0, edgecolor = 'none')

#adjust fonts
if plt.rcParams['font.family'] == ['serif']:
    plt.rcParams['font.serif'] = ['CMU Serif'] #'CMU Serif','dejavuserif'
    plt.rcParams['mathtext.fontset'] = 'cm' #supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
if plt.rcParams['font.family'] == ['sans-serif']:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] #select plotting font from ['DejaVu Sans','Bitstream Vera Sans','Computer Modern Sans Serif','Lucida Grande','Verdana','Geneva','Lucid','Arial','Helvetica','Avant Garde','sans-serif']
    plt.rcParams['mathtext.fontset'] = 'dejavusans' #supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']

#pass variables
cmap = plt.get_cmap(plt.rcParams['image.cmap']) #write colormap into variable
fig_size_single = plt.rcParams['figure.figsize']
fig_left = plt.rcParams['figure.subplot.left']
fig_right = plt.rcParams['figure.subplot.right']
fig_bottom = plt.rcParams['figure.subplot.bottom']
fig_top = plt.rcParams['figure.subplot.top']
fig_h = plt.rcParams['figure.subplot.hspace']
fig_w = plt.rcParams['figure.subplot.wspace']

#function to be called in other codes
def figure(fig_size='single',fig_type=''):
    if fig_size == 'single':
        fig_size = fig_size_single
    if fig_size == 'doublecolumn':
        fig_size = [fig_size_single[0]*2+0.5,fig_size_single[1]]
    if fig_size == 'doublerow':
        fig_size = [fig_size_single[0],fig_size_single[1]*2+0.5]  
    if fig_size == 'REM':
        fig_size = [fig_size_single[0]*2+0.5,fig_size_single[1]*5/6]
    plt.rc('figure', figsize = fig_size)
    plt.rc('figure.subplot', top=1-(1-fig_top)*fig_size_single[1]/fig_size[1], bottom=fig_bottom*fig_size_single[1]/fig_size[1], left=fig_left*fig_size_single[0]/fig_size[0], right=1-(1-fig_right)*fig_size_single[0]/fig_size[0])
    if fig_type == 'bar':
        plt.rc('axes', grid = False)
        plt.rc('xtick', labelsize = 'medium', top = True, bottom = True)
        plt.rc('ytick', labelsize = 'medium', left = True, right = True)
    elif fig_type == 'image':
        plt.rc('axes', grid = False)
        plt.rc('xtick', labelsize = 'medium', top = False, bottom = False)
        plt.rc('ytick', labelsize = 'medium', left = False, right = False)
    elif fig_type == 'LSM_analysis':
        plt.rc('axes', grid = False)
        plt.rc('legend', handletextpad = 0.3)
        plt.rc('xtick', labelsize = 'medium', top = False, bottom = False)
        plt.rc('ytick', labelsize = 'medium', left = False, right = False)
        plt.rc('figure.subplot', top = 0.9, bottom = 0.1, wspace = 0.15)
    elif fig_type == 'radar':
        plt.rc('axes', grid = False)
        plt.rc('xtick', labelsize = 'small', direction = 'in')
        plt.rc('ytick', direction = 'in')
        #plt.rc('xaxis', labellocation = 'right')
        plt.rc('figure.constrained_layout', use = False)
    elif fig_type == 'parallels':
        plt.rc('axes', grid = False)
        plt.rc('figure.subplot', top = 0.875)
        plt.rc('xtick', labelsize = 'medium', direction = 'in')
        plt.rc('ytick', direction = 'inout')
        #plt.rc('ytick', direction = 'out', labellocation = 'right')
        plt.rc('figure.constrained_layout', use = False)
    elif fig_type == 'LSM':
        #plt.rc('figure.subplot', top = 0.9, bottom = 0.1, left = 0.2, right = 0.8, hspace = 0.15, wspace = 0.25)
        plt.rc('figure.subplot', bottom = 0.1)
        plt.rc('axes', grid = False)
        plt.rc('xtick', labelsize = 'medium', direction = 'in', bottom = True)
        plt.rc('ytick', direction = 'in', left = True)
        plt.rc('figure.constrained_layout', use = False)
    elif fig_type == 'ProzInt':
        #plt.rc('figure.subplot', top = 0.9, bottom = 0.1, left = 0.2, right = 0.8, hspace = 0.15, wspace = 0.25)
        plt.rc('figure.subplot', bottom = 0.1)
        plt.rc('axes', grid = True)
        plt.rc('xtick', top = True, bottom = True, direction = 'in', labelbottom = True)
        plt.rc('ytick', right = True, left = True, direction = 'in', labelleft = True)
        plt.rc('figure.constrained_layout', use = False)
    elif fig_type == 'Lumentum':
        #plt.rc('figure.subplot', top = 0.95, bottom = 0.05)
        plt.rc('xtick', top = True, bottom = True, direction = 'in', labelbottom = True)
        plt.rc('ytick', right = True, left = True, direction = 'in', labelleft = True)
        plt.rc('figure.constrained_layout', use = False)
    elif fig_type == 'AnaGeo':
        #plt.rc('figure.subplot', top = 0.9, bottom = 0.1, left = 0.2, right = 0.8, hspace = 0.15, wspace = 0.25)
        plt.rc('figure.subplot', bottom = 0.1)
        plt.rc('axes', grid = False)
        plt.rc('xtick', top = True, bottom = True, direction = 'in', labelbottom = True)
        plt.rc('ytick', right = True, left = True, direction = 'in', labelleft = True)
        plt.rc('figure.constrained_layout', use = False)
    else: #re-add which might have been altered in previous figures
        plt.rc('axes', grid = True) 
        plt.rc('xtick', top = True, bottom = True, direction = 'in', labelbottom = True)
        plt.rc('ytick', right = True, left = True, direction = 'in', labelleft = True)
        plt.rc('figure.constrained_layout', use = False)