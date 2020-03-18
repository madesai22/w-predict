from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt

dataset = Dataset('omega_redo.nc')

lat = dataset['/Omega/latitude'][:]
lon = dataset['/Omega/longitude'][:]-360.0
w_all = dataset['/Omega/w'][:]
time_all = dataset['/Omega/time'][:]
levels = dataset['/Omega/depth'][:]

lat_line = 0#len(lat)-1 # index in latitude 
depth = [0,14,24]
_min = -10
_max = 10


hovmoller_surface = np.zeros((len(time_all),len(lon)))
hovmoller_mld = np.zeros((len(time_all),len(lon))) #mix layer depth
hovmoller_low = np.zeros((len(time_all),len(lon)))

for i in range(0,len(time_all)):
	hovmoller_surface[i] = w_all[i,lat_line,:,depth[0]]
	hovmoller_mld[i] = w_all[i,lat_line,:,depth[1]]
	hovmoller_low[i] = w_all[i,lat_line,:,depth[2]]

# convert date list
dt_list =[]
start_date = dt.date(1950,1,1)
for i in range(0,len(time_all)):
		td = dt.timedelta(seconds =time_all[i]*60*60)
		new_date = td+start_date
		dt_list.append(new_date.year)

fig, axes = plt.subplots(3, 1)
degree_sign= u'\N{DEGREE SIGN}'

top = cm.get_cmap('Blues_r', 128)
bottom = cm.get_cmap('Reds', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='RedBlue')

for ax in axes:
	ax.set_yticks(np.arange(0,23,5))
	ax.set_xticks(np.arange(0,len(time_all),53))
	ax.set_xticklabels(dt_list[0::53])
	ax.set_yticklabels(lon[0::5])
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
	p = ax.pcolormesh(np.transpose(hovmoller_surface),cmap = newcmp,vmin = _min, vmax = _max)
	for tick in ax.xaxis.get_majorticklabels():
		tick.set_fontsize(7) 

cb_ax = fig.add_axes([.92, 0.1, 0.02, 0.75])
cbar = fig.colorbar(p, cax=cb_ax,label='m/day')
#cbar.set_label("m/day")



axes[0].set_title("Surface (depth = 0m)")
axes[1].set_title("Mixed layer depth (depth = 150 m)")
axes[2].set_title("depth = 250 m")

axes[1].set_ylabel("Longitude ("+degree_sign+"W)")
axes[2].set_xlabel("Time (January)")

p = axes[0].pcolormesh(np.transpose(hovmoller_surface),cmap = newcmp,vmin = _min, vmax = _max)#,cmap = newcmp)
axes[1].pcolormesh(np.transpose(hovmoller_mld),cmap = newcmp,vmin = _min, vmax = _max)
axes[2].pcolormesh(np.transpose(hovmoller_low),cmap = newcmp,vmin = _min, vmax = _max)

plt.subplots_adjust(hspace=0.7)
fig.suptitle("Hovmoller diagrams at latitude = %f "% lat[lat_line]+ degree_sign +"N" )
plt.show()


	#print(w_date.shape)


