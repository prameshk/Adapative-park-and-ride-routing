'''
This file is created for preliminary plots in the paper such as analyzing travel time on different sections of the freeway
Created by
Pramesh Kumar
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


location = 'S:/Projects/Ridesharing Work/APNR/Data/'



# Plotting for the uncongested conditions
df = pd.read_csv(location + 'uncongestedDataNoPR.csv', encoding = "ISO-8859-1")
df['time'] = pd.to_datetime(df.time, unit='s').dt.strftime('%H:%M:%S').astype(str).values.tolist()





import seaborn as sns; sns.set()
plt.rc('font', family='serif', size = 25)
plt.rcParams['figure.figsize'] = 15, 5
Z = pd.pivot_table(df, values='travelTime',index='time',columns='fromNode')
#cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
#sns.palplot(sns.diverging_palette(0, 40, n=9))
ax = sns.heatmap(Z, cmap='RdYlGn_r', cbar_kws={'label': 'Travel time (sec)'}, vmin=0, vmax=75)
#ax = sns.heatmap(Z, cbar_kws={'label': 'travelTime'})
ax.invert_yaxis()
ax.set(xlabel='Towards downtown', ylabel='Clock')

plt.savefig('uTT.png', dpi=100)

plt.show()


# Plottinf for the congested conditions




df = pd.read_csv(location + 'congestedDataNoPR.csv', encoding = "ISO-8859-1")
df['time'] = pd.to_datetime(df.time, unit='s').dt.strftime('%H:%M:%S').astype(str).values.tolist()





import seaborn as sns; sns.set()
plt.rc('font', family='serif', size = 25)
plt.rcParams['figure.figsize'] = 15, 5
Z = pd.pivot_table(df, values='travelTime',index='time',columns='fromNode')
#cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
#sns.palplot(sns.diverging_palette(0, 40, n=9))
ax = sns.heatmap(Z, cmap='RdYlGn_r', cbar_kws={'label': 'Travel time (sec)'}, vmin=0, vmax=75)
#ax = sns.heatmap(Z, cbar_kws={'label': 'travelTime'})
ax.invert_yaxis()
ax.set(xlabel='Towards downtown', ylabel='Clock')

plt.savefig('cTT.png', dpi=100)

plt.show()
