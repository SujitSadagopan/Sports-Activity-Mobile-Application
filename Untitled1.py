
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_excel("C:/Users/sujit/Desktop/ump3.xlsx",skiprows=7)


# In[4]:



df.head()


# In[35]:


df.tail()


# In[7]:


df[['Latitude','Longitude']]


# In[5]:


x1 = df['Latitude']
    


# In[10]:


import osmapi
api = osmapi.OsmApi()
print(api.NodeGet(123))


# In[10]:


j = list(x1)


# In[11]:


k = list(y1)


# In[46]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[47]:


import cufflinks as cf


# In[9]:


j


# In[18]:


j[1:100]


# In[ ]:


average 


# In[48]:


init_notebook_mode(connected=True)


# In[37]:


from gmplot import gmplot

# Place map
gmap = gmplot.GoogleMapPlotter(-37.82, 144.98, 15)

# Polygon
lats, lons = j,k
gmap.plot(lats, lons, 'cornflowerblue', edge_width=1)

# Scatter points
#top_attraction_lats, top_attraction_lons = j,k
#gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=0.001, marker=False)

# Marker
hidden_gem_lat, hidden_gem_lon = -37.82, 144.98
gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')

# Draw
z = gmap.draw("my_maper.html")


# In[38]:


df.head()


# In[39]:


speed = df['Speed (km/h)']


# In[40]:


Time = df['Cumulative Time']


# In[41]:


energy = []


# In[42]:


energy = 60*(speed**2)


# In[45]:


energy


# In[48]:


import matplotlib as pyplot


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__)


# In[51]:


import cufflinks as cf


# In[52]:


init_notebook_mode(connected=True)


# In[61]:


cf.go_offline()


# In[62]:


df2


# In[54]:


df1 = pd.DataFrame(data = df)


# In[73]:


df2 = pd.DataFrame(data = enerlist)


# In[77]:


timelis


# In[74]:


enerlist = list(energy)


# In[76]:


timelis = list(Time)


# In[124]:


dftime = pd.DataFrame(data = timelis,columns= ['A'])


# In[126]:


dftime.head()


# In[138]:


dfener = pd.DataFrame(data = enerlist,columns= ['B'])


# In[139]:


j = dftime.rank(axis = 1)


# In[203]:


timeval = pd.DataFrame(data = r,columns=['A'])


# In[204]:


df3 = [timeval,dfener]


# In[207]:


findf = pd.concat(df3,axis=1)


# In[209]:


findf


# In[174]:


len(timelis)


# In[198]:


r = list(range(len(timelis)))


# In[199]:


len(r)


# In[200]:


k = 0
for i in timelis:
    hours = timelis[k].strftime('%H')
    minutes = timelis[k].strftime('%M')
    sec = timelis[k].strftime('%S')
    msec = timelis[k].strftime('%f')
    total_time = ((int(hours)*3600)+(int(minutes)*60)+int(sec)+(int(msec)/(1000000)))
    r[k] = total_time
    k = k+1


# In[201]:


r


# In[ ]:


timeval = pd.DataFrame(data = r,columns='A')


# In[183]:





# In[181]:





# In[176]:


int(et_program_hours)


# In[177]:


int(get_program_minutes)


# In[148]:


findf = pd.concat(df3,axis = 1)


# In[162]:


for i in timelis:
    i.split(':')
    


# In[210]:


findf[['A','B']].iplot(kind='spread')


# In[214]:


findf.iplot(kind='scatter',x='A',y='B',mode='markers',size=1)


# In[215]:


import seaborn as sns


# In[219]:


sns.jointplot(x='A',y='B',data=findf,kind='reg')


# In[231]:


findf['B'].iplot(kind='hist',bins=50)


# In[232]:


findf.iplot(kind='box')


# In[223]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[96]:


dftime


# In[81]:


dfener = pd.DataFrame(data = enerlist, index = 'A')


# In[82]:


dffinal  = [dftime,dfener]


# In[86]:


pd.concat(dffinal,join_axes=1)


# In[75]:


df2


# In[64]:


df3 = [df1,df2]


# In[65]:


dataf = pd.concat(df3)


# In[66]:


dataf


# In[19]:


z


# In[ ]:


import geocoder
g = geocoder.freegeoip('99.240.181.199')
g.json


# In[ ]:


l


# In[49]:


cf.go_offline()


# In[44]:


import numpy as np


# In[ ]:


j1 = np.split(l,0)


# In[50]:


kkr


# In[6]:


y1 = df['Longitude']


# In[ ]:


kkr = list(enumerate(l))


# In[ ]:


res_list = [x[0][0] for x in l]


# In[52]:


from mpl_toolkits.basemap import Basemap 


# In[53]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


# In[74]:


import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize


# In[75]:



m = Basemap(resolution='h', # c, l, i, h, f or None
            projection='merc',
            lat_0=-38, lon_0=145.36,
            llcrnrlon=145., llcrnrlat= -38, urcrnrlon=142, urcrnrlat=26)


# In[113]:



fig, ax = plt.subplots(figsize=(10,10))
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

m = Basemap(resolution='h', # c, l, i, h, f or None
            projection='lcc',width = 0.05E6,height = 0.05E6
            lat_0=-38, lon_0=145.36)

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()
#def plot_area(pos):
 #   count = new_areas.loc[new_areas.pos == pos]['count']
x, y = m(j,k)
#size = (count/1000) ** 2 + 3
m.plot(x, y, 'o', markersize=100, color='#444444', alpha=0.8)
    


# In[77]:



m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()


# In[115]:


fig = plt.figure(figsize=(15, 15))
m = Basemap(projection='lcc', resolution='f', 
            lat_0=-38, lon_0=145,
            width=0.05E6, height=0.05E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(k, j, latlon=True,
          cmap='Reds', alpha=0.5)


# In[ ]:


df3 = pd.DataFrame({'x':[1159332],'y':j,'z':k})
df3.iplot(kind='surface',colorscale='rdylbu')


# In[ ]:



from shapely.geometry import Point, Polygon


# In[ ]:


import geopandas as gpd
states = gpd.read_file('C:/Users/sujit/Desktop/ausdata.csv')
print(states.head())


# In[27]:


import geojsonio


# In[7]:


import os


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[120]:


import osmapi
api = osmapi.OsmApi()
print api.NodeGet(123)


# In[121]:


pdoc --html osmapi.OsmApi


# In[119]:


from ipyleaflet import Map
cordinates = [-37.8136, 144.9631]
zoom = 10
m = Map(center=cordinates,zoom = zoom)
m


# In[ ]:


k


# In[11]:


import gmplot

gmap = gmplot.GoogleMapPlotter(-38, 145, 16)

gmap.plot(j, k, 'cornflowerblue', edge_width=10)
gmap.scatter(j, k, '#3B0B39', size=40, marker=False)
#gmap.scatter(marker_lats, marker_lngs, 'k', marker=True)
#gmap.heatmap(heat_lats, heat_lngs)

gmap.draw("mymap.html")


# In[ ]:


import gmplot


# In[ ]:


import matplotlib as plt


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.jointplot(x='Latitude',y='Longitude',data=df,kind='scatter',size = 100)


# In[ ]:


plt.scatter(x[1],y[1])


# In[ ]:


gmap


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt

