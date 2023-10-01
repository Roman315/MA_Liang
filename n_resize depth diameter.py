
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 10})

fig,ax = plt.subplots(figsize=(4,3))
 
#Achsen von bis
ax.set_xlim(0, 800000)
ax.set_ylim(25, 50)

#x/y Werte
x_Werte = np.array([786432	,196608	,87381.33333,	49152	,31457.28	,21845.33333,	16049.63265	,12288])
y_Werte = np.array([31.47,	30.50	,29.70	,29.03	,28.30	,26.75,	27.03	,26.42])

x_Werte2 = np.array([786432	,196608	,87381.33333,	49152	,31457.28	,21845.33333,	16049.63265	,12288])
y_Werte2 = np.array([44.91,	44.42,	43.90,	42.46,	42.15,	41.40,	40.12,	40.91])



#Achsenbeschriftung
ax.set_xlabel('Auflösung in Pixel')
ax.set_ylabel('Tiefe / Durchmesser in μm') # µm


ax.scatter(x_Werte, y_Werte, label='Tiefe')
ax.scatter(x_Werte2, y_Werte2, label='Durchmesser')
#ax.scatter(x_Werte3, y_Werte3, label='centerofgravity')
#ax.scatter(x_Werte4, y_Werte4, label='Iteratione weightedleastsquares')

ax.plot(x_Werte, y_Werte, x_Werte2, y_Werte2)
ax.legend()
plt.tight_layout()
plt.savefig('n_resize depth diameter.png', dpi=300)
plt.show()