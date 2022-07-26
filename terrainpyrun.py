import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset,num2date
import datetime
import matplotlib as mpl

#plt.plot([1,2],[1,2])
#nameoffigure = "sefudeu"
#string_in_string = "{}".format(nameoffigure)
#plt.savefig(string_in_string)

rootgrp = Dataset('cm1out.nc','r')

dims = rootgrp.dimensions

vars = rootgrp.variables

attrs = rootgrp.ncattrs

ndims = len(dims)

print ('number of dimensions = ' + str(ndims))

for key in dims:
    print ('dimension['+key+'] = ' +str(len(dims[key])))

gattrs = rootgrp.ncattrs()
ngattrs = len(gattrs)

print ('number of global attributes = ' + str(ngattrs))

for key in gattrs:
    print ('global attribute['+key+']=' + str(getattr(rootgrp,key)))

vars = rootgrp.variables
nvars = len(vars)
print ('number of variables = ' + str(nvars))

for var in vars:
    print ('---------------------- variable '+var+'----------------')
    print ('shape = ' + str(vars[var].shape))
    vdims = vars[var].dimensions
    for vd in vdims:
        print ('dimension['+vd+']=' + str(len(dims[vd])))

#rain = vars['rain'][:]
#P= vars['prs'][:] #pressure
xh= vars['xh'][:] #x coordinate
yh= vars['yh'][:] #y cooordinate
xf= vars['xf'][:] #extra x coordinate
#yf= vars['yf'][:] #extra y coordinate
#z= vars['z'][:] #height 
z= vars['zh'][:] #height (use only for version 20.2 of cm1)
zh=vars['zhval'][:] #height on nominal levels (use for plots if terrain is not flat and in version 20.2)
#zh=vars['zh'][:] #height on nominal levels (use for plots if terrain is not flat)
u= vars['u'][:] #u wind
v= vars['v'][:] #v wind
w= vars['w'][:] #vertical velocity
#dbz= vars['dbz'][:] #reflectivity
time= vars['time'][:] #time 
theta= vars['th'][:] #potential temperature
thpert= vars['thpert'][:] #potential temperature perturbation
#B= vars['buoyancy'][:] #buoyancy
#zs= vars['zs'][:] #height of the terrain
#sw= vars['swten'][:] #heating from shortwaves (K/s)
#solrad= vars['swdnt'][:] #incoming solar radiation
#thpert= vars['thpert'][:] #potential temperature perturbation
cloud= vars['cldfra'][:] #cloud fraction
#mavail= vars['mavail'][:] #moisture availability
#lu0= vars['lu'][:] #subgrid tke
#xland= vars['xland'][:] #1 for land and 2 for water
#z0= vars['znt'][:] #surface roughness length
qv= vars['qv'][:] #mixing ratio
#tke= vars['xkzm'][:] #subgrid tke



#Make a datetime array
time1=[]
for k in range(0,len(time)):
    time1.append(datetime.datetime(2019, 6, 25, 6, 30, 0) + datetime.timedelta(seconds=int(time[k]))  )
    
time1=np.array(time1)

#Converts from seconds to readable time
def convert(timeinsecs):
   time = timeinsecs
   day = time // (24 * 3600)
   time = time % (24 * 3600)
   hour = time // 3600
   time %= 3600
   minutes = time // 60
   time %= 60
   seconds = time
   return("Day %d at %d:%d:%d" % (day, hour, minutes, seconds))

#Makes a readable time array (different from time1)
time2=[]
for k in range(0,len(time)):
    time2.append(convert(time[k]+1800+86400))
time2=np.array(time2)

#Function to help in the colorbar plotting
def custom_div_cmap(numcolors=26, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap

bwr_custom = custom_div_cmap(20)

bwr_custom_thpert = custom_div_cmap(30)


#Animation of U or V winds, potential temperature and pressure (xz section )
xm,zm=np.meshgrid(xh,z)
        
for k in range(0,len(time),1):
    
    
    fig=plt.figure(figsize=(10,10))
    
    
    ax=fig.add_subplot(2,1,1)
    wndspeed = np.sqrt(np.array(v[:,:,0:3,:])**2   +  np.array(u[:,:,0:3,:-1])**2)
  #  plt.contourf(xm,zm,wndspeed[k,:,0,:],np.arange(0,20.1,0.1),cmap='CMRmap')
    plt.pcolormesh(xm,zh[0,:,0,:],wndspeed[k,:,0,:],cmap='CMRmap',vmin=0, vmax=20)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])),name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])
    
    
    ax=fig.add_subplot(2,1,2)
    #plt.contourf(xm,zm,w[k,:-1,0,:],np.arange(-0.1,0.11,0.01),cmap='seismic')
    #plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    plt.pcolormesh(xm,zh[0,:,0,:],w[k,:-1,0,:],cmap=bwr_custom,vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Vertical velocity (m/s)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])
    
  
    #plt.pause(0.1)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/wndspdandvertmotion/'+string_in_string)
    plt.close()   

#Animation of U and V winds (xz section)
xm,zm=np.meshgrid(xh,z)

for k in range(0,len(time),1):


    fig=plt.figure(figsize=(10,10))


    ax=fig.add_subplot(2,1,1)
    plt.contourf(xm,zh[0,:,0,:],u[k,:,0,:-1],np.arange(-10,10.1,0.1),cmap='seismic')
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])


    ax=fig.add_subplot(2,1,2)
    plt.contourf(xm,zh[0,:,0,:],v[k,:,0,:],np.arange(-20,20.1,0.1),cmap='seismic')
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])

    #plt.pause(0.5)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/UandVwind/'+string_in_string)
    plt.close()




#Animation of perturbation potential temperature and cloud fraction (xz section )
xm,zm=np.meshgrid(xh,z)
        
        
for k in range(0,len(time),1):
    
    
    fig=plt.figure(figsize=(10,10))
   
    
    
    ax=fig.add_subplot(2,1,1)
    #plt.contourf(xm,zm,thpert[k,:,0,:],np.arange(-10,10.5,0.5),cmap='seismic')
    #plt.pcolormesh(xm,zm,thpert[k,:,0,:],cmap='seismic',vmin=-15, vmax=15)
    plt.pcolormesh(xm,zh[0,:,0,:],thpert[k,:,0,:],cmap=bwr_custom_thpert,vmin=-15, vmax=15)
    plt.colorbar(label='Potential temperature (K)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])),name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])
    
    ax=fig.add_subplot(2,1,2)
    #plt.contourf(xm,zm,cloud[k,:,0,:],cmap='seismic')
    plt.pcolormesh(xm,zh[0,:,0,:],cloud[k,:,0,:],cmap='seismic')
    plt.colorbar(label='Cloud Fraction (Pa)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])
   
  
    #plt.pause(0.5)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/thpertandcloudfraction/'+string_in_string)
    plt.close()   




#Animation of potential temperature and mixing ratio (xz section )
xm,zm=np.meshgrid(xh,z)


for k in range(0,len(time),1):


    fig=plt.figure(figsize=(10,10))



    ax=fig.add_subplot(2,1,1)
    plt.pcolormesh(xm,zh[0,:,0,:],theta[k,:,0,:],cmap='CMRmap_r',vmin=300.0, vmax=379.0)
    plt.colorbar(label='Potential temperature (K)')
#   plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])),name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])

    ax=fig.add_subplot(2,1,2)
    plt.pcolormesh(xm,zh[0,:,0,:],qv[k,:,0,:],cmap='CMRmap', vmin=0, vmax=0.0075)
    plt.colorbar(label='Mixing ratio (kg/kg)')
    plt.title(time2[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-2968,2968])

 


    #plt.pause(0.5)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/thetaandmixratio/'+string_in_string)
    plt.close()



#Creates a four panel plot of the potential temp and virtual potential temp
thetaV = theta * (1 + 0.61*qv )

for k in range(0,len(time),1):


    fig=plt.figure(figsize=(10,10))
    fig.suptitle(time2[k],name='Arial',weight='bold',size=20)

    ax=fig.add_subplot(2,2,1)
    plt.plot(theta[k,:,0,0],z)
    plt.title('Edges of domain' ,name='Arial',weight='bold',size=20)
    plt.xlabel('Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([290,400])
    ax.set_ylim([0,14])
    
    ax=fig.add_subplot(2,2,2)
    plt.plot(theta[k,:,0,int(len(xh)/2)],z)
    plt.title('Center of domain' ,name='Arial',weight='bold',size=20)
    plt.xlabel('Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([290,400])
    ax.set_ylim([0,14])
    
    ax=fig.add_subplot(2,2,3)
    plt.plot(thetaV[k,:,0,0],z)
    plt.title('Edges of domain' ,name='Arial',weight='bold',size=20)
    plt.xlabel('Virtual Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([290,400])
    ax.set_ylim([0,14])
    
    ax=fig.add_subplot(2,2,4)
    plt.plot(thetaV[k,:,0,int(len(xh)/2)],z)
    plt.title('Center of domain' ,name='Arial',weight='bold',size=20)
    plt.xlabel('Virtual Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([290,400])
    ax.set_ylim([0,14])

    
    plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0.5)
    #plt.pause(0.5)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/soundings/'+string_in_string)
    plt.close()
  


#Creates a profile of U and V winds

for k in range(0,len(time),1):


    fig=plt.figure(figsize=(10,10))
    fig.suptitle(time2[k],name='Arial',weight='bold',size=20)
    
    ax=fig.add_subplot(2,1,1)
    plt.plot(u[k,:,0,0],z)
    #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    plt.xlabel('U wind (m/s)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-10,20])
    ax.set_ylim([0,14])
    
    
    
    ax=fig.add_subplot(2,1,2)
    plt.plot(v[k,:,0,0],z)
    #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    plt.xlabel('V wind (m/s)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    ax.set_xlim([-10,20])
    ax.set_ylim([0,14])

    plt.subplots_adjust(bottom=0.15, top=0.85, hspace=0.5)
    #plt.pause(0.5)
    nameoffigure = time2[k]
    string_in_string = "{}".format(nameoffigure)
    plt.savefig('/glade/scratch/mgomes/cm1.2/runfigs/UandVprofile/'+string_in_string)
    plt.close()





















