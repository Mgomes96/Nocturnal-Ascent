#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset,num2date
import datetime
#from myfunctions import maxmin



#This part gets the parcel data but its not necessary

rootgrp = Dataset('/home/owner/Documents/LLJConvection/cm1model/cm1out_pdata.nc','r')

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
        

xparcel= vars['x'][:]
yparcel= vars['y'][:]
zparcel= vars['z'][:]


#%%
########################################################################################################################
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset,num2date
import datetime
#from myfunctions import maxmin

#This part reads the output from the model


rootgrp = Dataset('/home/owner/Documents/LLJConvection/cm1model/cm1out4.nc','r')

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

#133 
#172      
limit = 1000
#rain = vars['rain'][:limit]
#P= vars['prs'][:limit] #pressure
#Ppert= vars['prspert'][:limit] #pressure perturbation
#Pi= vars['pi'][:limit] #nondimensional pressure (exner function)
#Pipert= vars['pipert'][:limit] #nondimensional pressure perturbation (exner function

PGFpertwPi = vars['wb_pgrad'][:limit] #pressure gradient term in the CM1 w equation
Bw = vars['wb_buoy'][:limit] #buoyancy in the CM1 w equation
#hadvw = vars['wb_hadv'][:limit] #horizontal advection in the CM1 w equation
#vadvw = vars['wb_vadv'][:limit] #vertical advection in the CM1 w equation
# whturb = vars['wb_hturb'][:limit] #horizontal turbulence tendency in the CM1 w equation
# wvturb = vars['wb_vturb'][:limit] #vertical turbulence tendency in the CM1 w equation
# wrdamp = vars['wb_rdamp'][:limit] #rayleigh damping tendency in the CM1 w equation
# hidiffw = vars['wb_hidiff'][:limit] #horizontal diffusion term in the w equation
# vidiffw = vars['wb_vidiff'][:limit] #vertical diffusion term in the w equation

# PGFpertuPi = vars['ub_pgrad'][:limit] #pressure gradient term in the CM1 u equation
# fcoru = vars['ub_cor'][:limit] #coriolis term in the CM1 u equation
# urdamp = vars['ub_rdamp'][:limit] #rayleigh damping term in the CM1 u equation
# hadvu = vars['ub_hadv'][:limit] #horizontal advection in the CM1 u equation
# vadvu = vars['ub_vadv'][:limit] #vertical advection in the CM1 u equation
# uhturb = vars['ub_hturb'][:limit] #horizontal turbulence tendency in the CM1 u equation
# uvturb = vars['ub_vturb'][:limit] #vertical turbulence tendency in the CM1 u equation
# upblten = vars['ub_pbl'][:limit] #pbl tendency in the CM1 u equation
# uidiff = vars['ub_hidiff'][:limit] #Diffusion (incuding artificial) in the CM1 u equation

# PGFpertvPi = vars['vb_pgrad'][:limit] #pressure gradient term in the CM1 v equation
# fcorv = vars['vb_cor'][:limit] #coriolis term in the CM1 v equation
# vrdamp = vars['vb_rdamp'][:limit] #rayleigh damping term in the CM1 v equation
# hadvv = vars['vb_hadv'][:limit] #horizontal advection in the CM1 v equation
# vadvv = vars['vb_vadv'][:limit] #vertical advection in the CM1 v equation
# vhturb = vars['vb_hturb'][:limit] #horizontal turbulence tendency in the CM1 v equation
# vvturb = vars['vb_vturb'][:limit] #vertical turbulence tendency in the CM1 v equation
# vpblten = vars['vb_pbl'][:limit] #pbl tendency in the CM1 v equation
# vidiff = vars['vb_hidiff'][:limit] #Diffusion (incuding artificial) in the CM1 v equation


#th_hadv= vars['ptb_hadv'][:] #Horizontal advection of potential temperature


xh= vars['xh'][:] #x coordinateyy
yh= vars['yh'][:] #y cooordinate
xf= vars['xf'][:] #extra x coordinate
yf= vars['yf'][:] #extra y coordinate
#z= vars['z'][:limit] #height
z= vars['zh'][:] #height (use only for version 20.2 of cm1)
#zh=vars['zh'][:limit] #height on nominal levels (use for plots if terrain is not flat and in version 19.8)
zh=vars['zhval'][:limit] #height on nominal levels (use for plots if terrain is not flat and in version 20.2)
u= vars['u'][:limit] #u wind
#u= vars['ua'][:limit] #u wind (for restart runs only)
v= vars['v'][:limit] #v wind
#v= vars['va'][:limit] #v wind (for restart runs only)
w= vars['w'][:limit] #vertical velocity
#w= vars['wa'][:limit] #vertical velocity (for restart runs only)
#dbz= vars['dbz'][:limit] #reflectivity
time= vars['time'][:limit] #time 
theta= vars['th'][:limit] #potential temperature
#theta= vars['tha'][:limit] #potential temperature (for restart runs only)
#thpert= vars['thpert'][:limit] #potential temperature perturbation
#N= np.sqrt(vars['nm'][:limit]) #brunt-vaisala frequency 
B= vars['buoyancy'][:limit] #buoyancy
#rho= vars['rho'][:limit] #dry air density
#zs= vars['zs'][:limit] #height of the terrain
solrad= vars['swten'][:limit] #heating from shortwaves (K/s)
#swdnt= vars['swdnt'][:limit] #heating from shortwaves (K/s)
#solrad= vars['swdnt'][:limit] #incoming solar radiation
#thpert= vars['thpert'][:limit] #potential temperature perturbation
#cloud= vars['cldfra'][:limit] #cloud fraction
#mavail= vars['mavail'][:limit] #moisture availability
#lu0= vars['lu'][:limit] #subgrid tke
#xland= vars['xland'][:limit] #1 for land and 2 for water
#z0= vars['znt'][:limit] #surface roughness length
#qv= vars['qv'][:limit] #water vapor mixing ratio
# tke= vars['xkzm'][:limit] #subgrid tke
# kmh= vars['kmh'][:limit] #subgrid horizontal eddy viscosity (eddy diffusivity for momentum)
# khh= vars['khh'][:limit] #subgrid horizontal eddy diffusivity (eddy diffusivity for temperature)
# kmv= vars['kmv'][:limit] #subgrid vertical eddy viscosity (eddy diffusivity for momentum)
# khv= vars['khv'][:limit] #subgrid vertical eddy diffusivity (eddy diffusivity for temperature)






#Make a datetime array
time1=[]
for k in range(0,len(time)):
    time1.append(datetime.datetime(2019, 6, 25, 0, 30, 0) + datetime.timedelta(seconds=int(time[k]))  )
    
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
   return("Day %d at %d:%d" % (day, hour, minutes))





#Makes a readable time array (different from time1)
time2=[]
for k in range(0,len(time)):
    time2.append(convert(time[k]+1800+86400))
time2=np.array(time2)

#making a day/night array
daynight = np.ones_like(time2)
blackyellow = np.ones_like(time2)
for k in range(0,len(time2)):
    if "6:30" in time2[k] or "7:30" in time2[k] or "8:30" in time2[k] or "9:30" in time2[k] or "10:30" in time2[k] or "11:30" in time2[k] or "12:30" in time2[k] or "13:30" in time2[k] or "14:30" in time2[k] or "15:30" in time2[k]:
        daynight[k] = "\u263c"
        blackyellow[k] = "y"
        #print(time2[k])
    else:
        daynight[k] = "\u263e"
        blackyellow[k] = "k"
        

#%%

#Calculates the virtual potential temperature
#thetaV = theta * (1 + 0.61*qv )





##Print Pressure  (horizontal section)
#X=np.linspace(-59,59,60)
#Y=np.linspace(-59,59,60)
#
#xm,ym=np.meshgrid(X,Y)
#
#plt.contourf(xm,ym,P[5][0],cmap='CMRmap')
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('y axis')

#%%
#Prints potential temperature profile
plt.figure()
#plt.figtext(0.30, 0.90, "\u263c", fontsize='large', color='y', ha ='right')
# plt.title(r'U$_\max$ = 5 m $\rms^{-1}$        U$_\min$ = 5 m $\rms^{-1}$',x=0.5, y=1.02)
plt.rcParams.update({"font.size": 16})
plt.plot(theta[0,:,0,0],z,linewidth=3,color='k')
plt.xlabel("Potential Temperature (K)")
plt.ylabel('Height (km)')
#plt.ylim([0,4])
#plt.xlim([290,335])
plt.grid('True')

#%%
#Prints mixing ratio profile
plt.figure()
plt.rcParams.update({"font.size": 16})
plt.plot(qv[0,:,0,0],z,linewidth=3,color='k')
plt.xlabel(r'Mixing Ratio ($\rmkgkg^{-1}$) ')
plt.ylabel('Height (km)')
plt.ylim([0,14])
plt.xlim([0,0.002])
plt.grid('True')
plt.show()


#%%

#Plots panel of W like in Shapiro et al 2018
st_time = 89 # 67
intval = 2
xm,zm=np.meshgrid(xh,z)
fig=plt.figure(figsize=(10,10))

ax=fig.add_subplot(5,1,1)
plt.contourf(xm,zh[0,:,0,:]/1000.0,w[st_time,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic')
#plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
#plt.title(time2[k],name='Arial',size=20)
#plt.xlabel('X Domain (km)',name='Arial',size=16)
plt.ylabel('Height (km)',name='Arial',size=16)
ax.set_xlim([-500,500])
ax.set_ylim([0,6])
plt.tick_params(labelbottom = False, bottom = False)
plt.text(0.87, 0.9, time2[st_time], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax=fig.add_subplot(5,1,2)
plt.contourf(xm,zh[0,:,0,:]/1000.0,w[st_time+intval,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic')
#plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
#plt.title(time2[k],name='Arial',size=20)
#plt.xlabel('X Domain (km)',name='Arial',size=16)
plt.ylabel('Height (km)',name='Arial',size=16)
ax.set_xlim([-500,500])
ax.set_ylim([0,6])
plt.tick_params(labelbottom = False, bottom = False)
plt.text(0.87, 0.9, time2[st_time+intval], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax=fig.add_subplot(5,1,3)
plt.contourf(xm,zh[0,:,0,:]/1000.0,w[st_time+2*intval,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic')
#plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
#plt.title(time2[k],name='Arial',size=20)
#plt.xlabel('X Domain (km)',name='Arial',size=16)
plt.ylabel('Height (km)',name='Arial',size=16)
ax.set_xlim([-500,500])
ax.set_ylim([0,6])
plt.tick_params(labelbottom = False, bottom = False)
plt.text(0.87, 0.9, time2[st_time+2*intval], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax=fig.add_subplot(5,1,4)
plt.contourf(xm,zh[0,:,0,:]/1000.0,w[st_time+3*intval,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic')
#plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
#plt.title(time2[k],name='Arial',size=20)
#plt.xlabel('X Domain (km)',name='Arial',size=16)
plt.ylabel('Height (km)',name='Arial',size=16)
ax.set_xlim([-500,500])
ax.set_ylim([0,6])
plt.tick_params(labelbottom = False, bottom = False)
plt.text(0.87, 0.9, time2[st_time+3*intval], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

ax=fig.add_subplot(5,1,5)
plt.contourf(xm,zh[0,:,0,:]/1000.0,w[st_time+4*intval,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic')
#plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
#plt.title(time2[k],name='Arial',size=20)
plt.xlabel('X Domain (km)',name='Arial',size=16)
plt.ylabel('Height (km)',name='Arial',size=16)
ax.set_xlim([-500,500])
ax.set_ylim([0,6])
plt.text(0.87, 0.9, time2[st_time+4*intval], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

plt.subplots_adjust(bottom=0.07, top=0.93, hspace=0.15, right=0.8)
#fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(plt.contourf(xm,zh[0,:,0,:]/1000.0,w[94,:-1,0,:],np.arange(-0.05,0.055,0.005),cmap='seismic'), cax=cbar_ax,label=r'Vertical velocity (m $\rms^{-1}$)')



#%%




##Print U, V and W winds (horizontal section )
#xm,ym = np.meshgrid(xf,yh)
#
#xn,yn = np.meshgrid(xh,yf)
#
#xw,yw = np.meshgrid(xh,yh)
#
#
#
##plt.contourf(xm,ym,u[30,10,:,:],cmap='CMRmap')
##plt.contourf(xn,yn,v[30,10,:,:],np.arange(0,20,1),cmap='CMRmap')
#plt.contourf(xw,yw,w[50,10,:,:],cmap='seismic')
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('y axis')



##Print Reflectivity (horizontal section)
#X=np.linspace(-59,59,60)
#Y=np.linspace(-59,59,60)
#
#xm,ym=np.meshgrid(X,Y)
#
#plt.contourf(xm,ym,dbz[5][20],cmap='CMRmap')
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('y axis')




##Print U and V winds (xz section )
#xmv,zmv=np.meshgrid(xh,z)
##xmu,zmu=np.meshgrid(xf,z)
#
#
##plt.contourf(xmu,zmu,u[120,:,50,:],np.arange(-10,10,0.1),cmap='CMRmap')
##plt.colorbar()
##plt.contour(xmu,zmu,u[96,:,100,:],np.arange(0,20,1),colors='k')
#
#
#plt.contourf(xmv,zmv,v[90,:,1,:],np.arange(0,20,0.1),cmap='CMRmap')
#plt.colorbar()
##plt.contour(xmv,zmv,v[5,:,1,:],np.arange(0,20,1),colors='k')
#
#
#plt.xlabel('x axis')
#plt.ylabel('z axis')



#Animation of U or V winds (xz section )
#xm,zm=np.meshgrid(xh,z)
#
##Comment/uncomment this part for terrain following coordinates or not (dont use this anymore)
##for k in zmv:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##for k in zmu:
##    for t in range(0,len(k)-1):
##        k[t] = k[t] + zs[0,0,t]/1000.0
#        
#        
#        
#for k in range(0,len(v),5):
#    
#    
##    plt.contourf(xm,zh[0,:,0,:],u[k,:,1,:-1],np.arange(-10,10,0.1),cmap='seismic')
##    plt.colorbar()
#    #plt.contour(xmu,zmu,u[96,:,100,:],np.arange(0,20,1),colors='k')
#    
#    
#    plt.contourf(xm,zh[0,:,0,:],v[k,:,1,:],np.arange(0,20,0.1),cmap='CMRmap')
#    plt.colorbar(label='Wind Speed (m/s)')
#    #plt.contour(xmv,zmv,v[5,:,1,:],np.arange(0,20,1),colors='k')
#    
#    
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    
#    
#    plt.pause(0.5)
#    plt.clf()




#Print Potential temperature or potential temp perturbation (xz section)
#xm,zm=np.meshgrid(xh,z)
##
##
###Comment/uncomment this part for terrain following coordinates or not (dont use this anymore)
###for k in zm:
###    for t in range(0,len(k)):
###        k[t] = k[t] + zs[0,0,t]/1000.0
##
#ax = plt.gca()
#
#plt.contourf(xm,zh[0,:,0,:],theta[0,:,0,:],np.arange(290,330,0.5),cmap='CMRmap')
##plt.contourf(xm,zm,thpert[0,:,0,:],np.arange(0,8.1,0.1),cmap='CMRmap')
##plt.contourf(xm,zh,theta[0,:,0,:],np.arange(290,330,0.5),cmap='CMRmap')
#ax.set_xlim([-1000,1000])
##
###To be used fr testing only
###ttheta = abs( abs(xm*1000)-np.amax(xm)*1000 )/371062# + abs( zm*1000-np.amax(zm)*1000 )/5000.0
###ttheta = -(xm*1000)*8/1000000 - zm*1000*8/2000.0 + 8
###plt.contourf(xm,zm,ttheta,np.arange(0,8,0.1),cmap='CMRmap')
##
##
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('z axis')







#Print Pressure (xz section)
#xm,zm=np.meshgrid(xh,z)
#
##Comment/uncomment this part for terrain following coordinates or not (dont use this anymore)
##for k in zm:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
#
#
#
#plt.contourf(xm,zh[0,:,0,:],P[0,:,0,:],cmap='CMRmap')
#
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('z axis')






#Animation of potential temperature (or perturbation) (xz section )
#xm,zm=np.meshgrid(xh,z)
#
#
##Comment/uncomment this part for terrain following coordinates or not (dont use this anymore)
##for k in zm:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
#        
# 
#              
#for k in range(0,len(theta),5):
#    
#    
#    #plt.contourf(xm,zh[0,:,0,:],theta[k,:,0,:],np.arange(290,330,0.5),cmap='CMRmap')
#    #plt.colorbar()
#    
#    plt.contourf(xm,zh[0,:,0,:],theta[k,:,0,:]-theta[0,:,0,:],np.arange(-10,10,0.1),cmap='seismic')
#    plt.colorbar()
#    
#    
#    
#    
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    
#    
#    plt.pause(0.5)
#    plt.clf()





#Animation of U or V winds, potential temperature and pressure (xz section)
#xm,zm=np.meshgrid(xh,z)
#
##Comment/uncomment this part for terrain following coordinates or not (dont use this anymore)
##for k in zmv:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##for k in zmu:
##    for t in range(0,len(k)-1):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##for k in zm:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
#        
#        
#        
#for k in range(0,len(time),4):
#    
#    #plt.figure()
#    
#    plt.subplot(2,1,1)
#    plt.contourf(xm,zm,u[k,:,0,:-1],np.arange(-10,10.1,0.1),cmap='seismic')
#    plt.colorbar(label='Wind Speed (m/s)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##    plt.subplot(2,1,1)
##    plt.contourf(xm,zm,v[k,:,0,:],np.arange(0,21.1,0.1),cmap='CMRmap')
##    plt.colorbar(label='Wind Speed (m/s)')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    plt.subplot(2,1,2)
#    wndspeed = np.sqrt(np.array(v[:,:,0:3,:])**2   +  np.array(u[:,:,0:3,:-1])**2)
#    plt.contourf(xm,zm,wndspeed[k,:,0,:],np.arange(0,20.1,0.1),cmap='CMRmap')
#    plt.colorbar(label='Wind Speed (m/s)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    
##    plt.subplot(2,1,1)
##    plt.contourf(xm,zm,thpert[k,:,0,:],np.arange(-10,10.5,0.5),cmap='seismic')
##    plt.colorbar(label='Potential temperature (K)')
##    plt.title(time2[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##   
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zm,B[k,:,0,:],np.arange(-0.4,0.42,0.02),cmap='seismic')
##    plt.colorbar(label='Buoyancy ($s^{-2}$)')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zh[0,:,0,:],P[k,:,0,:]-P[0,:,0,:],np.arange(-350,350,10),cmap='seismic')
##    plt.colorbar(label='Presure Perturbaion (Pa)')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zm,cloud[k,:,0,:],cmap='seismic')
##    plt.colorbar(label='Cloud Fraction (Pa)')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zm,w[k,:-1,0,:],np.arange(-0.1,0.11,0.01),cmap='seismic')
##    plt.colorbar(label='Vertical velocity (m/s)')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
##    
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zm,qv[k,:,0,:],np.arange(0,0.0075,0.0001),cmap='CMRmap')
##    plt.colorbar(label='Water vapor mixing ratio')
##    plt.title(time2[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#
#    
#    plt.subplot(2,1,2)
#    plt.contourf(xm,zh[0,:,0,:],tke[k,:-1,0,:],np.arange(0,30,1),cmap='CMRmap')
#    plt.colorbar(label='Subgrid TKE')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
##    plt.subplot(2,1,2)
##    plt.contourf(xm,zm,PGFx[k,:,0,:],np.arange(-5,5.1,0.1),cmap='seismic')
##    plt.colorbar(label='Pressure gradient')
##    plt.title(time1[k],name='Arial',weight='bold',size=20)
##    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
##    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    
#  
#    plt.pause(0.5)
#    nameoffigure = time2[k]
#    string_in_string = "{}".format(nameoffigure)
#    plt.savefig(string_in_string)
#    plt.clf()   
    
    







#Print Reflectivity (xz section)
#xm,zm=np.meshgrid(xh,z)
#
#plt.contourf(xm,zm,dbz[190,:,1,:],cmap='CMRmap')
#
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('z axis')









#Print Pressure (xz section)
#xm,zm=np.meshgrid(xh,z)
#
#plt.contourf(xm,zm,P[0,:,50,:],np.arange(32473,100115,1000),cmap='CMRmap')
#
#
#plt.colorbar()
#
#plt.xlabel('x axis')
#plt.ylabel('z axis')









#Print V wind in function of time and height
#plt.title('v wind in function of time and height')
#zv,tv=np.meshgrid(z,time)
#plt.contourf(tv,zv,np.nanmean(v[:,:,:,:],axis=(2,3)),np.arange(0,20,1),cmap='CMRmap')
#plt.colorbar()
#
#
#plt.xlabel('time (seconds)')
#plt.ylabel('height (km)')




#%%
# time=time[0:85]
# time2=time2[0:85]
# zh=zh[0:85]
# u=u[0:85]
# v=v[0:85]
reducao = 50
#Print wind speed in function of time and height
fig = plt.figure()
plt.rcParams.update({"font.size": 16})
#plt.title('Wind Speed as a Function of Time and Height',weight='bold',name='Arial',size=20)
wndspeed = np.sqrt(np.array(v[:,:,0:3,:])**2   +  np.array(u[:,:,0:3,:-1])**2)
zv,tv=np.meshgrid(z,time[:len(time)-reducao])
#field = plt.contourf(tv,zv,np.nanmean(wndspeed[:,:,:,:],axis=(2,3)),np.arange(0,20,1),cmap='CMRmap')
#ufield = plt.contourf(tv,zv,np.nanmean(u[:,:,:,:],axis=(2,3)),np.arange(-10,10,1),cmap='CMRmap')
#field2 = plt.contourf(tv,zv,np.nanmean(wndspeed[:,:,:,164:165],axis=(2,3)),np.arange(0,20,1),cmap='CMRmap')
field2 = plt.contourf(tv,zv,np.nanmean(wndspeed[:len(time)-reducao,:,:,7],axis=(2)),np.arange(0,20,1),cmap='CMRmap')  #328
#field2terrain = plt.contourf(tv,zh[:,:,0,328]/1000,wndspeed[:,:,0,328],np.arange(0,21,1),cmap='CMRmap')
cbar = plt.colorbar()
cbar.set_label(r'Wind speed ($\rmms^{-1}$) ')
plt.xticks(time[0:len(time):6], time2[0:len(time):6], rotation='vertical')
plt.ylabel(r'Height (m) ')
plt.gcf().autofmt_xdate()
plt.ylim([0,10])
#plt.ylim([977/1000,4000/1000])
#plt.yticks(np.arange(0,5,1))
#plt.contour(field,np.arange(0,20,1),colors='k')
#plt.clabel(field,inline=False,fontsize=8,colors='k')
#plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
plt.show()

#%%
#Print wind speed u and v in function of time and height but with subplots
fig,ax1 = plt.subplots(figsize=(20,20)) 
plt.rcParams.update({'font.size':16})
#plt.title('Wind Speed as a Function of Time and Height',weight='bold',name='Arial',size=20)
wndspeed = np.sqrt(np.array(v[:,:,0:3,:])**2   +  np.array(u[:,:,0:3,:-1])**2)
zv,tv=np.meshgrid(z,time)

plt.subplot(3,1,1)
field2 = plt.contourf(tv,zv,np.nanmean(wndspeed[:,:,:,353:354],axis=(2,3)),np.arange(0,22,1),cmap='CMRmap')

cbar = plt.colorbar()
#cbar.set_label(r'U wind (m $\rms^{-1}$)', name='Arial',size=18)
plt.tick_params(labelbottom = False, bottom = False)
# plt.xticks(time[0:len(time):4], time2[0:len(time):4], rotation='vertical')
plt.ylabel('Height (km)',size=20)
# plt.gcf().autofmt_xdate()
plt.ylim([0,6])

plt.subplot(3,1,2)
field2 = plt.contourf(tv,zv,np.nanmean(u[:,:,:,353:354],axis=(2,3)),np.arange(-10,11,1),cmap='seismic')


cbar = plt.colorbar()
#cbar.set_label(r'V wind (m $\rms^{-1}$)', name='Arial',size=18)
plt.tick_params(labelbottom = False, bottom = False)
# plt.xticks(time[0:len(time):4], time2[0:len(time):4], rotation='vertical')
plt.ylabel('Height (km)',size=20)
# plt.gcf().autofmt_xdate()
plt.ylim([0,6])

plt.subplot(3,1,3)
field2 = plt.contourf(tv,zv,np.nanmean(v[:,:,:,353:354],axis=(2,3)),np.arange(-0,22,1),cmap='CMRmap')

cbar = plt.colorbar()
#cbar.set_label(r'Wind Speed (m $\rms^{-1}$)', name='Arial',size=18)
plt.xticks(time[0:len(time):3], time2[0:len(time):3], rotation='vertical',size=13)
plt.ylabel('Height (km)',size=20)
plt.gcf().autofmt_xdate()
plt.ylim([0,6])

plt.subplots_adjust(bottom=0.12, top=0.97, hspace=0.09)

plt.show()

nameoffigure = 'winds25Nbaroclinic+800km'
#nameoffigure = 'winds25Nbaroclinic-70km' 
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/owner/Documents/LLJConvection/cm1model/figures/"+string_in_string)

#%%
#Print w and theta in function of time and height but with subplots
reducao = 10
fig,ax1 = plt.subplots(figsize=(20,20)) 
plt.rcParams.update({'font.size':16})
#plt.title('Wind Speed as a Function of Time and Height',weight='bold',name='Arial',size=20)
wndspeed = np.sqrt(np.array(v[:,:,0:3,:])**2   +  np.array(u[:,:,0:3,:-1])**2)
zv,tv=np.meshgrid(z,time[:len(time)-reducao])

# plt.subplot(2,1,1)
# field3 = plt.contourf(tv,zv,np.nanmean(theta[:,:,:,319:320],axis=(2,3)),np.arange(290,340,2),cmap='CMRmap')

# cbar = plt.colorbar()
# #cbar.set_label(r'U wind (m $\rms^{-1}$)', name='Arial',size=18)
# plt.tick_params(labelbottom = False, bottom = False)
# # plt.xticks(time[0:len(time):4], time2[0:len(time):4], rotation='vertical')
# plt.ylabel('Height (km)',size=20)
# # plt.gcf().autofmt_xdate()
# plt.ylim([0,6])

plt.subplot(1,1,1)
field3 = plt.contourf(tv,zv,np.nanmean(w[:len(time)-reducao,:-1,:,319:320],axis=(2,3)),np.arange(-0.07,0.07,0.005),cmap='seismic')

cbar = plt.colorbar()
#cbar.set_label(r'V wind (m $\rms^{-1}$)', name='Arial',size=18)
plt.xticks(time[0:len(time)-reducao:6], time2[0:len(time)-reducao:6], rotation='vertical')
plt.ylabel('Height (km)',size=20)
plt.gcf().autofmt_xdate()
plt.ylim([0,6])


plt.subplots_adjust(bottom=0.12, top=0.97, hspace=0.09)

plt.show()

#nameoffigure = '45Nbaroclinic+90km'
nameoffigure = '45Nbaroclinic-70km'
string_in_string = "{}".format(nameoffigure)
plt.savefig("/home/owner/Documents/LLJConvection/cm1model/figures/"+string_in_string)

#%%
#Print ageostrophic wind speed in function of time and height
fig = plt.figure()
#plt.title('Ageostrophic Wind Speed in Function of Time and Height',name='Arial',weight='bold',size=20)
awndspeed = np.sqrt((v[:,:,0:3,:]-10.0)**2   +  (u[:,:,0:3,:-1])**2)
zv,tv=np.meshgrid(z,time)
field = plt.contourf(tv,zv,np.nanmean(awndspeed[:,:,:,:],axis=(2,3)),np.arange(0,9.5,0.2),cmap='inferno')
#field2 = plt.contourf(tv,zv,np.nanmean(awndspeed[:,:,:,165:170],axis=(2,3)),np.arange(0,9.5,0.2),cmap='inferno')
cbar = plt.colorbar()
cbar.set_label("Wind Speed ($ms^{-1}$)", name='Arial',size=18)
plt.xticks(time[0:len(time):5], time2[0:len(time):5], rotation='vertical')
plt.gcf().autofmt_xdate()
plt.ylim([0,4])
#plt.contour(field,np.arange(0,20,1),colors='k')
#plt.clabel(field,inline=False,fontsize=8,colors='k')

plt.ylabel('Height (km)',name='Arial',size=18,style='italic')
#%%

#Print potential temperature in function of time and height
#plt.title('Potential Temperature in Function of Time and Height',name='Arial',weight='bold',size=20)
#zv,tv=np.meshgrid(z,time)
#field = plt.contourf(tv,zv,np.nanmean(theta[:,:,:,:],axis=(2,3)),np.arange(280,340,1),cmap='CMRmap')
#plt.colorbar(label='Potential Temperature (K)')
#plt.xticks(time[0:len(time):19], time2[0:len(time):19], rotation='vertical')
#plt.gcf().autofmt_xdate()
##plt.contour(field,np.arange(0,20,1),colors='k')
##plt.clabel(field,inline=False,fontsize=8,colors='k')
#
#plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')



#Print vertical motion in function of time and height
# plt.figure()
# plt.title('Vertical motion in Function of Time and Height',name='Arial',weight='bold',size=20)
# zv,tv=np.meshgrid(z,time)
# field = plt.contourf(tv,zv,np.nanmean(w[:,:-1,:,159:160],axis=(2,3)),np.arange(-0.05,0.051,0.001),cmap='seismic')
# plt.colorbar(label='Vertical motion (m/s)')
# plt.xticks(time[6:len(time):3], time2[6:len(time):3], rotation='vertical')
# plt.gcf().autofmt_xdate()
# #plt.contour(field,np.arange(0,20,1),colors='k')
# #plt.clabel(field,inline=False,fontsize=8,colors='k')

# plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')










#Plots the height of the terrain
#plt.plot(xh,zs[0,1,:]/1000,color='k',linewidth=5)
#ax = plt.gca()
##ax.set_xticks(np.linspace(np.amin(xh),np.amax(xh),5))
##ax.set_ylim([0,9])
#plt.xlabel('X-Extent of Domain (km)',name='Arial',weight='bold',size=15,style='italic')
#plt.ylabel('Height (km)',name='Arial',weight='bold',size=15,style='italic')
#plt.grid(True)



#Plots the moisture avaliability 
#plt.plot(xh,mavail[0,1,:],color='k',linewidth=5)
#ax = plt.gca()
#plt.xlabel('X-Extent of Domain (km)',name='Arial',weight='bold',size=15,style='italic')
#plt.ylabel('Moisture Avaliability (km)',name='Arial',weight='bold',size=15,style='italic')
#plt.grid(True)



#Plots the u and v wind with height at a certain point
#plt.plot(v[150,:,0,1100],z,label='V wind')
#plt.plot(u[150,:,0,1100],z,label='U wind')
#plt.xlabel('Wind speed (m/s)')
#plt.ylabel('height (km)')
#ax = plt.gca()
#ax.set_xlim([0,20])
#ax.set_xticks(np.arange(0,21,1))
#ax.set_ylim([0,9])
#ax.set_yticks(np.arange(0,10,1))
#plt.grid(True)
#plt.legend()


#Ignore this
#ax = plt.gca()
#plt.plot(time1,[1]*len(time1))
#plt.gcf().autofmt_xdate()
#ax.xaxis.set_major_locator(plt.MaxNLocator(10))



#Plots the U wind at certain times
# fig=plt.figure(figsize=(10,10))
# ax=fig.add_subplot(1,1,1)
# #fig.suptitle("U Wind ",name='Arial',weight='bold',size=20)
# #plt.title('U Wind Profiles',name='Arial',weight='bold',size=20)
# plt.plot(u[70,:,0,0],z,label='22:30 local time')
# plt.plot(u[72,:,0,0],z,label='00:30 local time')
# plt.plot(u[74,:,0,0],z,label='02:30 local time')
# plt.plot(u[76,:,0,0],z,label='04:30 local time')
# plt.xlabel('U wind (m/s)',name='Arial',weight='bold',size=16,style='italic')
# plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
# plt.legend()
# ax.set_xlim([-10,20])
# ax.set_ylim([0,2])
    

#%%
#Plots the u and v wind ratio with time at a certain heights
plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(20,20))
xposition = 0
init_time = 44  
final_time = 68  
sunrise_time = 53  #was 54 for old 40km run
zposition1 = 44 #height1  was 19 for old 40km run
zposition2 = 94 #height2  was 44 for old 40km run
#plt.title('LLJ Winds at Distinct Heights',name='Arial',weight='bold',size=20)
plt.plot(u[init_time:final_time + 1,zposition1,0,xposition],v[init_time:final_time + 1,zposition1,0,xposition],label='900m',linestyle = ':',c='r')
plt.plot(u[init_time:final_time + 1,zposition2,0,xposition],v[init_time:final_time + 1,zposition2,0,xposition],label='1900m',linestyle = ':',c='b')

#plt.plot(u[init_time,0:100,0,xposition],v[init_time,0:100,0,xposition],label='ratao',c='k')

plt.plot(u[init_time,zposition1,0,xposition],v[init_time,zposition1,0,xposition],label='20:30 LST',color='white',marker='*',markersize=25,markerfacecolor='purple')
plt.plot(u[init_time + 8,zposition1,0,xposition],v[init_time + 8,zposition1,0,xposition],label='04:30 LST',color='white',marker='*',markersize=25,markerfacecolor='green')
plt.plot(u[init_time + 16,zposition1,0,xposition],v[init_time + 16,zposition1,0,xposition],label='12:30 LST',color='white',marker='*',markersize=25,markerfacecolor='crimson')
plt.plot(u[init_time + 24,zposition1,0,xposition],v[init_time + 24,zposition1,0,xposition],label='20:30 LST',color='white',marker='*',markersize=25,markerfacecolor='k')
plt.plot(u[sunrise_time,zposition1,0,xposition],v[sunrise_time,zposition1,0,xposition],label='Sunrise',color='white',marker='*',markersize=25,markerfacecolor='orange')

plt.plot(u[init_time,zposition2,0,xposition],v[init_time,zposition2,0,xposition],color='white',marker='*',markersize=25,markerfacecolor='purple')
plt.plot(u[init_time + 8,zposition2,0,xposition],v[init_time + 8,zposition2,0,xposition],color='white',marker='*',markersize=25,markerfacecolor='green')
plt.plot(u[init_time + 16,zposition2,0,xposition],v[init_time + 16,zposition2,0,xposition],color='white',marker='*',markersize=25,markerfacecolor='crimson')
plt.plot(u[init_time + 24,zposition2,0,xposition],v[init_time + 24,zposition2,0,xposition],color='white',marker='*',markersize=25,markerfacecolor='k')
plt.plot(u[sunrise_time,zposition2,0,xposition],v[sunrise_time,zposition2,0,xposition],color='white',marker='*',markersize=25,markerfacecolor='orange')

plt.plot([0],[10],label='Geostrophic wind',color='white',marker='*',markersize=25,markerfacecolor='y')
plt.xlabel('U wind ($ms^{-1}$)',name='Arial',size=18,style='italic')
plt.ylabel('V wind ($ms^{-1}$)',name='Arial',size=18,style='italic')
# plt.xlim([-5,5])
# plt.ylim([5,15])


#plt.xlim([-4,4])
#plt.ylim([5,14])
plt.legend(loc = 1)
plt.grid(True)

#%%

#Plots the u and v wind ratio with time at a certain heights (better version)
plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(20,20))
xposition = 0
init_time = 0  
final_time = 130  
sunrise_time = 53  #was 54 for old 40km run
zposition1 = 29 #height1  was 19 for old 40km run
zposition2 = 94 #height2  was 44 for old 40km run


# x = u[init_time:final_time + 1,zposition1,0,xposition]
# y = v[init_time:final_time + 1,zposition1,0,xposition]

# plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, color="k")

#-----------------------------------------------------------------------
init_time1 = 0  
final_time1 = 23

x1 = u[init_time1:final_time1 + 1,zposition1,0,xposition]
y1 = v[init_time1:final_time1 + 1,zposition1,0,xposition]
plt.quiver(x1[:-1], y1[:-1], x1[1:]-x1[:-1], y1[1:]-y1[:-1], scale_units='xy', angles='xy', scale=1, color="k",label="Day1")

#----------------------------------------------------------------------
init_time2 = 23  
final_time2 = 47 

x2 = u[init_time2:final_time2 + 1,zposition1,0,xposition]
y2 = v[init_time2:final_time2 + 1,zposition1,0,xposition]
plt.quiver(x2[:-1], y2[:-1], x2[1:]-x2[:-1], y2[1:]-y2[:-1], scale_units='xy', angles='xy', scale=1, color="blue",label="Day2")

#----------------------------------------------------------------------
init_time3 = 47  
final_time3 = 71 

x3 = u[init_time3:final_time3 + 1,zposition1,0,xposition]
y3 = v[init_time3:final_time3 + 1,zposition1,0,xposition]
plt.quiver(x3[:-1], y3[:-1], x3[1:]-x3[:-1], y3[1:]-y3[:-1], scale_units='xy', angles='xy', scale=1, color="red",label="Day3")

#----------------------------------------------------------------------
init_time4 = 71  
final_time4 = 95 

x4 = u[init_time4:final_time4 + 1,zposition1,0,xposition]
y4 = v[init_time4:final_time4 + 1,zposition1,0,xposition]
plt.quiver(x4[:-1], y4[:-1], x4[1:]-x4[:-1], y4[1:]-y4[:-1], scale_units='xy', angles='xy', scale=1, color="yellow",label="Day4")

#----------------------------------------------------------------------
init_time5 = 95  
final_time5 = 111 

x5 = u[init_time5:final_time5 + 1,zposition1,0,xposition]
y5 = v[init_time5:final_time5 + 1,zposition1,0,xposition]
plt.quiver(x5[:-1], y5[:-1], x5[1:]-x5[:-1], y5[1:]-y5[:-1], scale_units='xy', angles='xy', scale=1, color="green",label="Day5")



plt.plot([0],[10],label='Geostrophic wind',color='white',marker='*',markersize=25,markerfacecolor='y')
plt.xlabel('U wind ($ms^{-1}$)',name='Arial',size=18,style='italic')
plt.ylabel('V wind ($ms^{-1}$)',name='Arial',size=18,style='italic')
# plt.xlim([-5,5])
# plt.ylim([5,15])
plt.axis('equal')

#plt.xlim([-4,4])
#plt.ylim([5,14])
plt.legend(loc = 1)
plt.grid(True)




#%%   

############################################################################## 
    
#Plots the solar radiation
plt.rcParams.update({"font.size": 16})
time_len = 24
plt.figure()
plt.plot(time[0:time_len],solrad[0:time_len,209,0,0],linewidth=3,color='k')
plt.plot(time[5],solrad[5,209,0,0],label='Sunrise',color='white',marker='*',markersize=20,markerfacecolor='orange')
plt.plot(time[20],solrad[20,209,0,0],label='Sunset',color='white',marker='*',markersize=20,markerfacecolor='red')
plt.xticks(time[0:time_len:1], time2[0:time_len:1], rotation='vertical')
plt.gcf().autofmt_xdate()
plt.ylabel('Heating from shortwaves at the top of the atmosphere ($Ks^{-1}$)',name='Arial',size=20,style='italic')
#plt.xlim([0,20])
plt.grid('True')
plt.legend(fontsize=20)
#plt.plot(time,swdnt[:,0,0])


#%%
#Plots the soil characteristics
x= np.arange(-2000,2001,10)
MAVAIL = 0.3 - 0.23 * np.exp(-x**2/500**2)
ALBEDO = 0.15 + 0.07 * np.exp(-x**2/500**2)
THC = 0.04 + 0.015 * np.exp(-x**2/500**2)

plt.rcParams.update({"font.size": 16})
plt.plot(x,MAVAIL,linewidth=3,color='b',label='Moisture Availability')
plt.plot(x,ALBEDO,linewidth=3,color='k',label='Albedo')
plt.plot(x,THC,linewidth=3,color='r',label='Thermal Inertia')
plt.grid('True')
plt.legend(fontsize=20)
plt.xlabel('X Domain (km)',name='Arial',size=20,style='italic')

 




    
#%%
############################################################################## 
    
#This part creates a new sounding from output data
#
#
#    
##Choosing the time index we are using to extract our profile
#timeindex = 50
#    
#    
##Transforming the arrays into lists to make it easier 
#zexp = []
#for k in z:
#    zexp.append(k*1000.0)
#    
#thetaexp = []
#for k in theta[timeindex,:,0,0]:
#    thetaexp.append(k)
#    
#    
#mixratioexp = []
#for k in qv[0,:,0,0]:
#    mixratioexp.append(k)
#    
#
#uwndexp = []
#for k in u[0,:,0,0]:
#    uwndexp.append(k)
#    
#vwndexp = []
#for k in v[0,:,0,0]:
#    vwndexp.append(k)
#    
#
#    
#
#
##Creating the variables that are going into the sounding in string form
#
#for k in range(0,len(zexp)):
#    
#    while len(str(zexp[k])) < 20:
#        zexp[k] = str(zexp[k]) + ' '
#    
#    while len(str(thetaexp[k])) < 20:
#        thetaexp[k] = str(thetaexp[k]) + ' '
#        
#    while len(str(mixratioexp[k])) < 20:
#        mixratioexp[k] = str(mixratioexp[k]) + ' '
#        
#    while len(str(uwndexp[k])) < 20:
#        uwndexp[k] = str(uwndexp[k]) + ' '
#        
#    while len(str(vwndexp[k])) < 20:
#        vwndexp[k] = str(vwndexp[k]) + ' '
#        
##Creating the surface values for each souding variable
#p0 = 1000.0
#
#theta0 = theta[timeindex,0,0,0]
#
#mixratio0 = 0.5       
#
#        
#        
##writing the new sounding 
#f = open('input_sounding.txt', 'w' )
#
#f.write(str(p0)+ '     '+ str(theta0)+ '     ' + str(mixratio0) + '     ' + '\n')
#
#for k in range(0,len(zexp)):
#    f.write(zexp[k]  + thetaexp[k] + mixratioexp[k] + uwndexp[k]  + vwndexp[k] + '\n'  )
#
#
#f.close() 



#%%
#Creates a sounding for cm1

#Test lists only
#p0 = 1000.0
#
#theta0 = 299.0
#
#mixratio0 = 11.0
#
#z = np.array([100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0])
#
#theta = np.array([300.0,301.4,307.2,310.0,314.3,320.1,323.1,330.9])
#
#mixratio = np.array([10.0,8.0,7.8,7.2,8.0,1.0,0.5,0.2,0.1])
#
#uwnd = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
#
#vwnd = np.array([10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
#
#
#
#f = open('input_sounding.txt', 'w' )
#
#f.write(str(p0)+ '     '+ str(theta0)+ '     ' + str(mixratio0) + '     ' + '\n')
#
#for k in range(0,len(theta)):
#    f.write(str(z[k]) + '     ' + str(theta[k]) + '     ' + str(mixratio[k]) + '     ' + str(uwnd[k]) + '     ' + str(vwnd[k]) + '\n'  )
#
#
#f.close() 



#This code reads and plots the data from a cm1 sounding 
import re

f = open('/home/owner/Documents/LLJConvection/cm1model/input_sounding', 'r' )
r = f.readlines()

f.close()

for k in range(0,len(r)):
    r[k] = re.sub('\s+', ' ', r[k])
    r[k] = r[k].split(' ')
    
    
p0 = float(r[0][1])

theta0 = float(r[0][2])

mixratio0 = float(r[0][3])

z=[]
theta=[]
mixratio=[]
uwnd=[]
vwnd=[]


for k in range(1,len(r)):
    z.append(float(r[k][1]))
    theta.append(float(r[k][2]))
    mixratio.append(float(r[k][3]))
    uwnd.append(float(r[k][4]))
    vwnd.append(float(r[k][5]))
    
    
#plt.plot(theta,z)
plt.plot(mixratio,z)


#Generating a "fake" theta profile
thetaexp = list(np.ones_like(theta)*[0])

##Using linear function
#for k in range(0,len(thetaexp)):
#    thetaexp[k] = z[k] * 0.004 + 300


#Using exponential function
for k in range(0,len(thetaexp)):
    thetaexp[k] = 300 * np.exp(z[k]/75000) + np.exp((z[k]-2000)/3200)

#Using another exponential function
#for k in range(0,len(thetaexp)):
#    thetaexp[k] = 600 - 300 * np.exp(-(1.0/(30000.0**2))*(z[k]-0.0)**2)
    
#plt.plot(thetaexp,z)


#Multiplying the mixingratio by a certain number
#mixratio = list(np.array(mixratio)*6)
#mixratio0 = mixratio0*6

#Changing the low level moisture
# mixratio = np.array(mixratio)
# for k in range(0,len(mixratio)):
#     if k < 15:
#         mixratio[k] = mixratio[k]*6
# mixratio0 = mixratio0*6

mixratio = np.array(mixratio)
for k in range(0,len(mixratio)):
    #mixratio[k] = 1 * np.exp(abs(z[k]-14200)/27000) - 1 #+ np.exp((abs(z[k]-14000)-14000)/3200)
    #mixratio[k] = 0.2 * np.exp(-z[k]/1000) +0.005
    #mixratio[k] = 0.0025 + 0.6 * (14275-z[k])/14275
    #mixratio[k] = 0.001 + 0.5 * (14275-z[k])/14275  #real nice one
    #mixratio[k] = 0.001 + 0.3 * (14275-z[k])/14275
    #mixratio[k] = 0.001 + 0.2 * (14275-z[k])/14275  #noclouds
    #mixratio[k] = 0.001 + 0.25 * (14275-z[k])/14275  # no clouds at all
    #mixratio[k] = 0.0025 + 0.25 * (14275-z[k])/14275  #1st try
    #mixratio[k] = 0.003 + 0.3 * (14275-z[k])/14275  #2nd try
    #mixratio[k] = 0.001 + 0.4 * (14070-z[k])/14070 #3rd try
    #mixratio[k] = 0.001 + 0.3 * (14070-z[k])/14070 #4th try
    #mixratio[k] = 0.001 + 0.25 * (14070-z[k])/14070 #5th try  use this as base
    #mixratio[k] = 0.001 + 0.28 * (14070-z[k])/14070   #1st try
    #mixratio[k] = 0.002 + 0.28 * (14070-z[k])/14070   #0st try
    #mixratio[k] = 0.0025 + 0.28 * (14070-z[k])/14070   #0.5
    #mixratio[k] = 0.003 + 0.28 * (14070-z[k])/14070  #boa tbm
    #mixratio[k] = 1 * np.exp(abs(z[k]-14200)/70000) - 1
    #mixratio[k] = 0.000003 * z[k]
    #mixratio[k] = 1*np.exp(-z[k]/5000)  
    #mixratio[k] = 0.7*np.exp(-z[k]/5000)
    # mixratio[k] = 1*np.exp(-z[k]/4000)    #rbest one so far
    #mixratio[k] = 1*np.exp(-z[k]/3800)   #ayy
    #mixratio[k] = 1.2*np.exp(-z[k]/3800)  #ayy2
    #mixratio[k] = 1.3*np.exp(-z[k]/3800)   #ayy3
    #mixratio[k] = 1.4*np.exp(-z[k]/3800)    #ayy4
    #mixratio[k] = 1.4*np.exp(-z[k]/3500)     #ayy5
    mixratio[k] = 1.4*np.exp(-z[k]/3200)    #ayyyyyyyy
    #mixratio[k] = 5*np.exp(-z[k]/3200)
mixratio0 = mixratio[0]

# mixratio[k] = 5 * np.exp(-z[k]/1000)  


plt.plot(mixratio,z)
plt.show()
mixratio = list(mixratio)





    

#Creating new arrays to alow for creating the new sounding


for k in range(0,len(z)):
    while len(str(z[k])) < 20:
        z[k] = str(z[k]) + ' '
    
    while len(str(thetaexp[k])) < 20:
        thetaexp[k] = str(thetaexp[k]) + ' '
        
    while len(str(theta[k])) < 20:
        theta[k] = str(theta[k]) + ' '
        
    while len(str(mixratio[k])) < 30:
        mixratio[k] = str(mixratio[k]) + ' '
        
    while len(str(uwnd[k])) < 20:
        uwnd[k] = str(uwnd[k]) + ' '
        
    while len(str(vwnd[k])) < 20:
        vwnd[k] = str(vwnd[k]) + ' '
        

        
        
#writing the new sounding 
f = open('/home/owner/Documents/LLJConvection/cm1model/input_sounding_new', 'w' )

f.write(str(p0)+ '     '+ str(theta0)+ '     ' + str(mixratio0) + '     ' + '\n')

for k in range(0,len(z)):
    f.write(z[k]  + theta[k] + mixratio[k] + uwnd[k]  + vwnd[k] + '\n'  )


f.close() 

#%%







#############################################################################




   
#Animation of U or V winds and potential temperature (xz section )
#xmv,zmv=np.meshgrid(xh,z)
#xmu,zmu=np.meshgrid(xf,z)
#xm,zm=np.meshgrid(xh,z)
##
###Comment/uncomment this   part for terrain following coordinates or not
##for k in zmv:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##for k in zmu:
##    for t in range(0,len(k)-1):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##for k in zm:
##    for t in range(0,len(k)):
##        k[t] = k[t] + zs[0,0,t]/1000.0
##        
##        
#        
#for k in range(0,len(time),5):
#    
#    
#    plt.figure()
#    
#    plt.subplot(1,2,1)
#    plt.contourf(xmu,zmu,u[k,:,1,:],np.arange(-10,10,0.1),cmap='seismic')
#    plt.colorbar(label='Wind Speed (m/s)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    
#    plt.contourf(xmv,zmv,v[k,:,1,:],np.arange(0,20,0.1),cmap='CMRmap')
#    plt.colorbar(label='Wind Speed (m/s)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    plt.subplot(1,2,2)
#    plt.contourf(xm,zm,theta[k,:,0,:],np.arange(290,330,0.5),cmap='CMRmap')
#    plt.colorbar(label='Potential temperature (K)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    
#    plt.contourf(xm,zm,theta[k,:,0,:]-theta[0,:,0,:],np.arange(-10,10,0.1),cmap='seismic')
#    plt.colorbar(label='Potential temperature perturbation (K)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
##    
##    
##    
##    
##    
##    
##    
#    plt.pause(0.5)
#    plt.clf()    


####################################################################################################################################

#Creating the plots of maximum intensity and height of w


#This plot only has the max w and max height for each day
# maxheight = []
# maxtime =[]
# maxintensity = []
# for k in range (0,len(w)-24,24):
#     maxtime.append(np.where(w[k:k+24,:,:,:] == np.amax(w[k:k+24,:,:,:]))[0][0] + k)
#     maxheight.append(np.where(w[k:k+24,:,:,:] == np.amax(w[k:k+24,:,:,:]))[1][0])
#     maxintensity.append(np.amax(w[k:k+24,:,:,:]))

# time2plot=[]
# zplot =[]    
# for k in range(0,len(maxtime)):
#     time2plot.append(time2[maxtime[k]])
#     zplot.append(z[maxheight[k]])
    

# fig,ax1=plt.subplots()
# plt.xticks(maxtime, time2plot, rotation='vertical')
# plt.gcf().autofmt_xdate()
# plt.plot(maxtime,zplot,marker='*')
# ax1.set_ylim([0,3])
# plt.xlabel('Peak w time',name='Arial',weight='bold',size=16,style='italic')
# plt.ylabel('Peak height',name='Arial',weight='bold',size=16,style='italic')

# ax2=ax1.twinx()
# plt.plot(maxtime,maxintensity,marker='o',markersize='5')
# plt.ylabel('Max w',name='Arial',weight='bold',size=16,style='italic')




#This is a continuous plot of w and max height
# maxheight2 = []
# maxtime2 =[]
# maxintensity2 = []
# for k in range(0,len(w)):
#     maxtime2.append(k)
#     maxheight2.append(np.where(w[k,0:75,:,:] == np.amax(w[k,0:75,:,:]))[0][0])
#     maxintensity2.append(np.amax(w[k,:,:,:]))

# time2plot2=[]
# zplot2 =[]    
# for k in range(0,len(maxtime2)):
#     time2plot2.append(time2[maxtime2[k]])
#     zplot2.append(z[maxheight2[k]])
    

# fig,ax1=plt.subplots()
# plt.xticks(maxtime2[0:len(maxtime2):24], time2plot2[0:len(time2plot2):24], rotation='vertical')
# plt.gcf().autofmt_xdate()
# plt.plot(maxtime2,zplot2)
# #ax1.set_ylim([0,3])
# plt.xlabel('Peak w time',name='Arial',weight='bold',size=16,style='italic')
# plt.ylabel('Peak height',name='Arial',weight='bold',size=16,style='italic')

#%%
# This is the scatterplot of w with height and intensity
maxheight2 = []
maxtime2 =[]
maxintensity2 = []
timesteps = 101
maxheight = 166
for k in range(0,timesteps):
    maxtime2.append(k)
    maxheight2.append(np.where(w[k,0:maxheight,:,:] == np.amax(w[k,0:maxheight,:,:]))[0][0])
    maxintensity2.append(100*np.amax(w[k,:,:,:]))

time2plot2=[]
zplot2 =[]    
for k in range(0,len(maxtime2)):
    time2plot2.append(time2[maxtime2[k]])
    zplot2.append(z[maxheight2[k]]) 
    

maxintensity2.insert(0,4)
zplot2.insert(0,0.000000001)
maxtime2.insert(0,0)

time2plot3 = np.ones_like(time2plot2)
for k in range(0,len(time2plot2)):
    time2plot3[k] = time2plot2[k][9:len(time2plot2[k])]

fig,ax1=plt.subplots()
plt.rcParams.update({"font.size": 16})
#plt.xticks(maxtime2[1:len(maxtime2):8], time2plot2[0:len(time2plot2):8], rotation='vertical')
plt.xticks(maxtime2[1:len(maxtime2):6], time2plot3[0:len(time2plot3):6], rotation='vertical')
plt.gcf().autofmt_xdate()
plt.plot(maxtime2,zplot2,zorder=1)
plt.scatter(maxtime2,zplot2,c=maxintensity2,s=60,cmap='Spectral_r',zorder=2)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20)
cbar.set_label(r'Peak w intensity (cm $\rms^{-1}$)',size=20)
#ax1.set_ylim([0,3])
#plt.xlabel('hi',name='Arial',weight='bold',size=20,style='italic')
plt.ylabel('Peak height (km)',name='Arial',size=20)
ax1.set_xlim([0,timesteps-1])
ax1.set_ylim([0,4])
plt.grid(True)
plt.show()
    
#%%    

###############################################################################################

#Calculating and plotting the total kinetic energy of the system 
wndspeed = np.sqrt((v[:,:,0:3,:])**2   +  (u[:,:,0:3,:-1])**2)

zweight = np.ones_like(wndspeed)
for k in range(1,len(z)):
    zweight[:,k,:,:] = abs(z[k] - z[k-1]) * 1000
zweight[:,0,:,:] = z[0] * 1000

kenergy = np.ones_like(wndspeed)

kenergy = wndspeed**2 * rho * zweight/2

meankenergy = []
for k in kenergy:
    meankenergy.append(np.mean(k))

plt.figure()
plt.plot(time,meankenergy)


###############################################################################################

#%%
#Calculating the maximum vertical parcel displacements for every 24 hour period
reference = -99999999
for k in range (0,len(zparcel[0])):
    for t in range(0,len(time)-21+2-4-2-8):
        counter = 0
        while counter < 19:
            if abs(zparcel[t][k]-zparcel[t+1+counter][k]) > reference:
                max_displacement = abs(zparcel[t][k]-zparcel[t+1+counter][k])
                initial_time_index = t
                final_time_index = t+1+counter
                parcel_index = k       
                initial_x_position = xparcel[t][k]
                final_x_position = xparcel[t+1+counter][k]
                initial_z_position = zparcel[t][k]
                final_z_position = zparcel[t+1+counter][k]
                reference = abs(zparcel[t][k]-zparcel[t+1+counter][k])
                
            counter = counter + 1
    print (k)
        
print ("max_displacement = ", max_displacement) 
print ("initial_time = ", time2[initial_time_index])       
print ("final_time = ", time2[final_time_index])
print ("initial_x_position = ", initial_x_position)
print ("final_x_position = ", final_x_position)
print ("initial_z_position = ", initial_z_position)
print ("final_z_position = ", final_z_position)
        


    
#%%
#Plot of tke and 'pbl' tendency before and after sunset

fig=plt.figure(figsize=(20,20))
plt.rcParams.update({"font.size": 16})

twopm = 38
b4sunset = 43

ax=fig.add_subplot(2,2,1)
plt.plot(tke[twopm,:-1,0,0],z, linewidth=3,color='b')
#plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
plt.xlabel('Subgrid TKE ($m^{-2}s^{-2}$)',name='Arial',size=16,style='italic')
plt.ylabel('Height (km)',name='Arial',size=19,style='italic')
ax.set_xlim([0.0,270])
ax.set_ylim([0,4])
plt.grid(True)


ax=fig.add_subplot(2,2,2)
plt.plot(tke[b4sunset,:-1,0,0],z, linewidth=3,color='b')
#plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
plt.xlabel('Subgrid TKE ($m^{-2}s^{-2}$)',name='Arial',size=16,style='italic')
plt.ylabel('Height (km)',name='Arial',size=19,style='italic')
ax.set_xlim([0.0,270])
ax.set_ylim([0,4])
plt.grid(True)


ax=fig.add_subplot(2,2,3)
plt.plot(upblten[twopm,:,0,0],z, linewidth=3,color='b')
plt.xlabel('PBL tendency ($ms^{-2}$)',name='Arial',size=16,style='italic')
plt.ylabel('Height (km)',name='Arial',size=19,style='italic')
ax.set_xlim([-0.0006,0.0006])
plt.xticks(np.arange(-0.0006,0.0007,0.0003))
ax.set_ylim([0,4])
plt.grid(True)


ax=fig.add_subplot(2,2,4)
plt.plot(upblten[b4sunset,:,0,0],z, linewidth=3,color='b')
plt.xlabel('PBL tendency ($ms^{-2}$)',name='Arial',size=16,style='italic')
plt.ylabel('Height (km)',name='Arial',size=19,style='italic')
ax.set_xlim([-0.0006,0.0006])
plt.xticks(np.arange(-0.0006,0.0007,0.0003))
ax.set_ylim([0,4])
plt.grid(True)

#%%
#Calculating the buoyancy gradient 

dBdx=np.ones_like(B[:,:,:,:])*np.nan
for k in range(1,len(xh)-1):
    dBdx[:,:,:,k] = ( B[:,:,:,k+1] - B[:,:,:,k-1] ) / abs( (xh[2]-xh[0])*1000.0 )
    
    
dBdx5=np.ones_like(B[:,:,:,:])*np.nan
for k in range(1,len(xh)-10):
    dBdx5[:,:,:,k] = ( B[:,:,:,k+5] - B[:,:,:,k-5] ) / abs( (xh[2]-xh[0])*5000.0 )
    
#Calculating the buoyancy gradient during the hottest time of day only and only below a certain height
height1 = 25
height2 = 50
B_aftnoon = B[[17,17+24,17+48,17+72],height1:height2,:,:]
dBdx_aftnoon=np.ones_like(B_aftnoon[:,:,:,:])*np.nan
for k in range(1,len(xh)-1):
    dBdx_aftnoon[:,:,:,k] = ( B_aftnoon[:,:,:,k+1] - B_aftnoon[:,:,:,k-1] ) / abs( (xh[2]-xh[0])*1000.0 )
    

B_aftnoon5 = B[[17,17+24,17+48,17+72],height1:height2,:,:]
dBdx_aftnoon5=np.ones_like(B_aftnoon5[:,:,:,:])*np.nan
for k in range(1,len(xh)-5):
    dBdx_aftnoon5[:,:,:,k] = ( B_aftnoon5[:,:,:,k+5] - B_aftnoon5[:,:,:,k-5] ) / abs( (xh[2]-xh[0])*5000.0 )
    

#Calculating the buoyancy gradient 1h befor max w only below a certain height
height1 = 25
height2 = 50
B_night = B[[21,21+24,21+48,21+72],height1:height2,:,:]
dBdx_night=np.ones_like(B_night[:,:,:,:])*np.nan
for k in range(1,len(xh)-1):
    dBdx_night[:,:,:,k] = ( B_night[:,:,:,k+1] - B_night[:,:,:,k-1] ) / abs( (xh[2]-xh[0])*1000.0 )
    
B_night5 = B[[21,21+24,21+48,21+72],height1:height2,:,:]
dBdx_night5=np.ones_like(B_night5[:,:,:,:])*np.nan
for k in range(1,len(xh)-5):
    dBdx_night5[:,:,:,k] = ( B_night5[:,:,:,k+5] - B_night5[:,:,:,k-5] ) / abs( (xh[2]-xh[0])*5000.0 )
    
    



#%%
#Calculating the convergence/divergence (dudx only) 
dudx = np.ones_like(u)*np.nan


for k in range(1,len(xh)-1):
    dudx[:,:,:,k] = (u[:,:,:,k+1] - u[:,:,:,k-1]) / (xh[2]-xh[0])
    

#%%    
##################################################################################################
    
#Calculating the actual temperature
#T = np.ones_like(theta)*np.nan

# T = theta * (    100000.0 * ( P**(-1) )     )**(-2.0/7)

###############################################################################################

#Calculating the terms in the the dw/dt equation (navier stokes) for local change

#Calculating the local change term

dwdt=np.ones_like(w[:,:-1,:,:])*np.nan
for k in range(1,len(time)-1):
    dwdt[k,:,:,:] = ( w[k+1,:-1,:,:] - w[k-1,:-1,:,:] ) / abs(time[k-1]-time[k+1])


# #Getting the PGF in the w direction
# PGFw=np.ones_like(P)*np.nan
# for k in range(1,len(z)-1):
#     PGFw[:,k,:,:] = (rho[:,k,:,:])**(-1) * ( P[:,k+1,:,:] - P[:,k-1,:,:] ) / abs( (z[k-1]-z[k+1])*1000.0 )
    
# #Getting the PGFpert in the w direction
# PGFpertw=np.ones_like(Ppert)*np.nan
# for k in range(1,len(z)-1):
#     PGFpertw[:,k,:,:] = - (rho[:,k,:,:])**(-1) * ( Ppert[:,k+1,:,:] - Ppert[:,k-1,:,:] ) / abs( (z[k-1]-z[k+1])*1000.0 )
    
    
# #Getting the nondimensional PGFpert in the w direction
# PGFPipertw=np.ones_like(Pipert)*np.nan
# for k in range(1,len(z)-1):
#     PGFPipertw[:,k,:,:] = - (rho[:,k,:,:])**(-1) * ( Pipert[:,k+1,:,:] - Pipert[:,k-1,:,:] ) / abs( (z[k-1]-z[k+1])*1000.0 )
    

# #Getting the buoyancy term 
# Bw = Bw

# #Getting the first advection term 
# dwdz=np.ones_like(w[:,:-1,:,:])*np.nan
# for k in range(1,len(z)-1):
#     dwdz[:,k,:,:] = ( w[:,k+1,:,:] - w[:,k-1,:,:] ) / abs( (z[k-1]-z[k+1])*1000.0 )
    
# advwz = w[:,:-1,:,:] * dwdz

# #Getting the second advection term
# dwdx=np.ones_like(w[:,:-1,:,:])*np.nan
# for k in range(1,len(xh)-1):
#     dwdx[:,:,:,k] = ( w[:,:-1,:,k+1] - w[:,:-1,:,k-1] ) / abs( (xh[2]-xh[0])*1000.0 )
    
# advwx = u[:,:,:,:-1] * dwdx


# advtotal = advwx + advwz


###############################################################################################



#Calculating the terms in the the du/dt equation (navier stokes) for local change

dudt=np.ones_like(u[:,:,:,:-1])*np.nan
for k in range(1,len(time)-1):
    dudt[k,:,:,:] = ( u[k+1,:,:,:-1] - u[k-1,:,:,:-1] ) / abs(time[k-1]-time[k+1])




###############################################################################################


# #Calculating the du/dt by looking at the u equation terms outputted by cm1
# initPGF = np.ones_like(u[:,:,:,:]) * 0.0 + (0.000084 * 10.0)

# #dudtcm1 = urdamp + hadvu  + uhturb + uvturb + uidiff  + fcoru - initPGF

# dudtcm1 = urdamp + hadvu + vadvu + uhturb + uvturb + upblten + uidiff + fcoru + (PGFpertuPi - initPGF)

# ucm1=np.ones_like(u[:,:,:,:])*np.nan

# for k in range(0,len(time)-1):
#     ucm1[k+1,:,:,:] = u[k,:,:,:] + abs(time[0]-time[1])*dudtcm1[k,:,:,:] 
    

#Calculating the du/dt by looking at the u equation terms outputted by cm1 (in another way)
initPGF = np.ones_like(u[:,:,:,:]) * 0.0 + (0.000084 * 10.0)

# cor1 = v[:,:,:,:] * 0.000084 

# dudtcm1 =    - initPGF[:,:,:,:-1] + cor1[:,:,:-1,:]

# dudtcm1 = urdamp + hadvu + vadvu + uhturb + uvturb + uidiff + fcoru + (PGFpertuPi - initPGF)

# ucm1=np.ones_like(u[:,:,:,:])*np.nan

# for k in range(0,len(time)-1):
#     ucm1[k+1,:,:,:-1] = u[k,:,:,:-1] + abs(time[0]-time[1])*dudtcm1[k,:,:,:]

#Calculating the dw/dt by looking at the w equation terms outputted by cm1

# dwdtcm1 = PGFpertwPi + Bw + hadvw + vadvw + whturb + wvturb + wrdamp + hidiffw + vidiffw

# wcm1=np.ones_like(w[:,:,:,:])*np.nan

# for k in range(0,len(time)-1):
#     wcm1[k+1,:,:,:] = w[k,:,:,:] + abs(time[0]-time[1])*dwdtcm1[k,:,:,:] 



wcm1 = np.ones_like(w[:,:,:,:])*[0]
for t in range(0,len(w)-1):
    wcm1[t+1] = wcm1[t] + (PGFpertwPi[t] + Bw[t])*360 



#%%    
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

#Function to help in the colorbar plotting v2
def custom_div_cmap2(numcolors=26, name='custom_div_cmap2',
                    mincol='white', midcol='gray', maxcol='black'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name,
                                              colors =[mincol, midcol, maxcol],
                                              N=numcolors)
    return cmap

bwr_custom2 = custom_div_cmap2(20)

bwr_custom_thpert2 = custom_div_cmap2(30)
    
    
    
    

#Animation of U or V winds, potential temperature and pressure (xz section)
defasagem = 0
xm,zm=np.meshgrid(xh,z)

for k in range(0,len(time)-defasagem,1):

    
    fig=plt.figure(figsize=(10,10))
    plt.rcParams.update({"font.size": 16})
    fig.suptitle(time2[k],name='Arial',size=20)
    plt.figtext(0.30, 0.940, daynight[k], fontsize=50, color=blackyellow[k], ha ='right')
    #plt.rcParams.update({"font.size": 16})
    #xposition = 0
    xposition = 328
    xposition2 = 310


    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,u[k,:,0,:-1],np.arange(-10,11,1),cmap='seismic')
    # plt.colorbar(label=r'U wind (m $\rms^{-1}$)')
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-4000,4000])
    # ax.set_ylim([0,4])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,v[k,:,0,:],np.arange(-19,20,1),cmap='seismic')
    # plt.colorbar(label=r'V wind (m $\rms^{-1}$)')
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-4000,4000])
    # ax.set_ylim([0,4])
    
    # #wind speed
    # ax=fig.add_subplot(1,1,1)
    # wndspeed = np.sqrt(np.array(v[:,:,0:2,:])**2   +  np.array(u[:,:,0:2,:-1])**2)
    # plt.contourf(xm,zm,wndspeed[k,:,0,:],np.arange(0,20.1,0.1),cmap='CMRmap')
    # #plt.pcolormesh(xm,zm,wndspeed[k,:,0,:],cmap='CMRmap',vmin=0, vmax=8)
    # plt.colorbar(label='Wind Speed (m/s)')
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.title(time2[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    # #convergence
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,dudx[k,:,0,:-1],np.arange(-0.2,0.205,0.005),cmap='seismic')
    # #plt.pcolormesh(xm,zm,dudx[k,:,0,:-1],cmap='seismic',vmin=-0.3, vmax=0.3)
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,dudx[k,:,0,:-1],np.arange(-0.2,0.205,0.005),cmap='seismic')
    # plt.colorbar(label=r'Divergence ($\rms^{-1}$)')
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # #plt.title(time2[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',size=16,style='italic')
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,7])

    # ax=fig.add_subplot(2,1,1)
    # #plt.contourf(xm,zm,thpert[k,:,0,:],np.arange(-10,10.5,0.5),cmap='seismic')
    # #plt.pcolormesh(xm,zm,thpert[k,:,0,:],cmap='seismic',vmin=-15, vmax=15)
    # plt.pcolormesh(xm,zm,thpert[k,:,0,:],cmap=bwr_custom_thpert,vmin=-15, vmax=15)
    # #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,thpert[k,:,0,:],cmap=bwr_custom_thpert,vmin=-15, vmax=15)
    # plt.colorbar(label='Potential Temperature Perturbation (K)')
    # #plt.title(time2[k],name='Arial',weight='bold',size=20)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])),name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # #plt.contourf(xm,zm,theta[k,:,0,:],np.arange(290,330,1),cmap='Reds')
    # #plt.pcolormesh(xm,zm,theta[k,:,0,:],vmin=290, vmax=350,cmap=bwr_custom_thpert2)
    # #plt.contourf(xm,zm,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,theta[k,:,0,:],np.arange(290,330,1),cmap='Reds')
    # plt.colorbar(label='Potential temperature (K)')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,4])
    

  
    # ax=fig.add_subplot(2,1,1)
    # #plt.contourf(xm,zm,B[k,:,0,:],np.arange(-0.5,0.52,0.02),cmap='seismic')
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,B[k,:,0,:],np.arange(-0.5,0.52,0.02),cmap='seismic')
    # #plt.pcolormesh(xm,zm,B[k,:,0,:],cmap=bwr_custom_thpert,vmin=-0.4, vmax=0.4)
    # #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,B[k,:,0,:],cmap=bwr_custom_thpert,vmin=-0.4, vmax=0.4)
    # plt.colorbar(label=r'Buoyancy (m $\rms^{-2}$)')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)')
    # plt.ylabel('Height (km)')
    # ax.set_xlim([-4000,4000])
    # ax.set_ylim([0,4])

#    plt.subplot(2,1,2)
#    plt.contourf(xm,zh[0,:,0,:],P[k,:,0,:]-P[0,:,0,:],np.arange(-350,350,10),cmap='seismic')
#    plt.colorbar(label='Presure Perturbaion (Pa)')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')

    # plt.subplot(2,1,2)
    # plt.contourf(xm,zm,cloud[k,:,0,:],cmap='seismic')
    # plt.colorbar(label='Cloud Fraction (Pa)')
    # plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')

    #vertical motion
    ax=fig.add_subplot(2,1,1)
    plt.contourf(xm,zh[0,:,0,:]/1000.0,w[k,:-1,0,:],np.arange(-0.1,0.11,0.01),cmap='seismic')
    #plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.05, vmax=0.05)
    #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
    #plt.title(time2[k],name='Arial',size=20)
    plt.xlabel('X Domain (km)',name='Arial',size=16)
    plt.ylabel('Height (km)',name='Arial',size=16)
    #ax.set_xlim([-2968,2968])
    ax.set_xlim([-1000,1000])
    ax.set_ylim([0,7])
    
    # #vertical motion cm1
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,wcm1[k,:-1,0,:],np.arange(-0.1,0.11,0.01),cmap='seismic')
    # #plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.05, vmax=0.05)
    # #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    # plt.colorbar(label=r'Vertical velocity (m $\rms^{-1}$)')
    # #plt.title(time2[k],name='Arial',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,4])

#    plt.subplot(2,1,2)
#    plt.contourf(xm,zm,qv[k,:,0,:],np.arange(0,0.0075,0.0005),cmap='CMRmap')
#    plt.colorbar(label='Water vathpor mixing ratio')
#    plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')

    # ax=fig.add_subplot(2,2,1)
    # plt.contourf(xm,zm,tke[k,:-1,0,:],np.arange(0,10.5,0.2),cmap='CMRmap')
    # plt.colorbar(label='Subgrid TKE')
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(1,1,1)
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,dBdx[k,:,0,:],np.arange(-1.3964576*10**-6,1.3964576*10**-6 + 1.3964576*10**-6/20,1.3964576*10**-6/20),cmap='seismic')
    # #plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.05, vmax=0.05)
    # #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    # plt.colorbar(label=r'Buoyancy Gradient (m $\rms^{-1}$)')
    # #plt.title(time2[k],name='Arial',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,14])
    
    # #Potential temperature advection term
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zh[0,:,0,:]/1000.0,th_hadv[k,:,0,:],np.arange(-0.0002,0.0002 + 0.0002/20,0.0002/20),cmap='seismic')
    # #plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.05, vmax=0.05)
    # #plt.pcolormesh(xm,zh[0,:,0,:]/1000.0,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    # plt.colorbar(label=r'Advection of potential temperature (m $\rms^{-1}$)')
    # #plt.title(time2[k],name='Arial',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # #ax.set_xlim([-2968,2968])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,14])

    
    
    
    
#     ax=fig.add_subplot(2,1,1)
#     plt.contourf(xm,zm,PGFx[k,:,0,:],cmap='seismic')
#     plt.colorbar(label='Pressure gradient')
#     plt.title(time1[k],name='Arial',weight='bold',size=20)
#     plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic') 
#     plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
# #    ax.set_xlim([-2968,2968])
    
    
    # ax=fig.add_subplot(2,2,2)
    # plt.plot(u[k,:,0,xposition],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # # plt.title(time2[k],name='Arial',size=20)
    # plt.xlabel(r'U wind (m $\rms^{-1}$) ')
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # ax.set_xlim([-10,20])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    
    # ax=fig.add_subplot(2,2,1)
    # plt.plot(u[k,:,0,xposition2],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'U wind (m $\rms^{-1}$) ')
    # plt.ylabel('Height (km)')
    # ax.set_xlim([-10,20])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    # ax=fig.add_subplot(2,2,2)
    # plt.plot(v[k,:,0,xposition],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'V wind (m $\rms^{-1}$)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # ax.set_xlim([-10,20])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    # ax=fig.add_subplot(1,1,1)
    # wndspeed = np.sqrt(np.array(v[:,:,0:2,:])**2   +  np.array(u[:,:,0:2,:-1])**2)
    # plt.plot(wndspeed[k,:,0,xposition],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'Wind speed (m $\rms^{-1}$)',name='Arial',size=16)
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([0,20])
    # ax.set_ylim([0,5])
    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(ucm1[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('U wind cm1 (m/s)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-10,20])
    # ax.set_ylim([0,14])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(upblten[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('PBL tendency ($s^{-2}$)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-0.0015,0.0015])
    # ax.set_ylim([0,5])
    
    # ax=fig.add_subplot(2,2,4)
    # plt.plot(upblten[k,:,0,xposition],z,label='Turbulence')
    # plt.plot(fcoru[k,:,0,xposition] - initPGF[k,:,0,xposition],z,label='Coriolis minus base PGF')
    # plt.plot(PGFpertuPi[k,:,0,xposition],z,label='PPGF')
    # plt.plot(urdamp[k,:,0,xposition],z,label='Rayleigh Damping')
    # plt.plot(hadvu[k,:,0,xposition],z,label='Horizontal advection')
    # plt.plot(vadvu[k,:,0,xposition],z,label='Vertical advection')
    # plt.plot(uidiff[k,:,0,xposition],z,label='Artificial Diffusion')
    # # plt.plot(uhturb[k,:,0,xposition],z,label='Horizontal Turbulence')
    # # plt.plot(uvturb[k,:,0,xposition],z,label='Vertical Turbulence')
    # plt.legend(fontsize=13)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'Terms in the U equation of motion (m $\rms^{-2}$)')
    # plt.ylabel('Height (km)')
    # ax.set_xlim([-0.0015,0.0015])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    # ax=fig.add_subplot(2,2,3)
    # plt.plot(upblten[k,:,0,xposition2],z,label='Turbulence')
    # plt.plot(fcoru[k,:,0,xposition2] - initPGF[k,:,0,xposition],z,label='Coriolis free-atmosphere base PGF')
    # plt.plot(PGFpertuPi[k,:,0,xposition2],z,label='PPGF')
    # plt.plot(urdamp[k,:,0,xposition2],z,label='Rayleigh Damping')
    # plt.plot(hadvu[k,:,0,xposition2],z,label='Horizontal advection')
    # plt.plot(vadvu[k,:,0,xposition2],z,label='Vertical advection')
    # plt.plot(uidiff[k,:,0,xposition2],z,label='Artificial Diffusion')
    # # plt.plot(uhturb[k,:,0,xposition],z,label='Horizontal Turbulence')
    # # plt.plot(uvturb[k,:,0,xposition],z,label='Vertical Turbulence')
    # #plt.legend(fontsize=10)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'Terms in the U equation of motion (m $\rms^{-2}$)')
    # plt.ylabel('Height (km)',name='Arial',size=16)
    # ax.set_xlim([-0.0015,0.0015])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    
    # ax=fig.add_subplot(2,2,4)
    # plt.plot(vpblten[k,:,0,xposition],z,label='PBL tendency')
    # plt.plot(fcorv[k,:,0,xposition],z,label='Coriolis minus base PGF')
    # plt.plot(PGFpertvPi[k,:,0,xposition],z,label='PPGF')
    # plt.plot(vrdamp[k,:,0,xposition],z,label='Rayleigh Damping')
    # plt.plot(hadvv[k,:,0,xposition],z,label='Horizontal advection')
    # plt.plot(vadvv[k,:,0,xposition],z,label='Vertical advection')
    # plt.plot(vidiff[k,:,0,xposition],z,label='Artificial Diffusion')
    # # plt.plot(uhturb[k,:,0,xposition],z,label='Horizontal Turbulence')
    # # plt.plot(uvturb[k,:,0,xposition],z,label='Vertical Turbulence')
    # plt.legend(fontsize=13)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel(r'Terms in the V equation of motion (m $\rms^{-2}$)')
    # plt.ylabel('Height (km)')
    # ax.set_xlim([-0.0015,0.0015])
    # ax.set_ylim([0,4])
    # plt.grid(True)
    
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.plot(w[k,:-1,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('W wind',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-1.5,1.5])
    # ax.set_ylim([0,14])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(wcm1[k,:-1,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('W wind cm1 (m/s)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-1.5,1.5])
    # ax.set_ylim([0,14])

    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.plot(v[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('V wind (m/s)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-10,20])
    # ax.set_ylim([0,2])

    # ax=fig.add_subplot(2,1,2)
    # plt.plot(vpblten[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('PBL tendency ($s^{-2}$)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-0.0008,0.0008])
    # ax.set_ylim([0,14])

    # ax=fig.add_subplot(2,1,2)
    # plt.plot(dudtcm1[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('dudtcm1 ($s^{-2}$)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-0.00052,0.0004])
    # ax.set_ylim([0,2])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(dudt[k,:,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('dudt ($s^{-2}$)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-0.00052,0.0004])
    # ax.set_ylim([0,2])


    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(tke[k,:-1,0,0],z)
    # #plt.title(time2[k] + '     ' +str(int(solrad[k,0,0])) ,name='Arial',weight='bold',size=20)
    # plt.xlabel('Subgrid TKE ($s^{-2}$)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([0.0,150])
    # ax.set_ylim([0,3])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.plot(theta[k,:,0,0],z)
    # plt.title('Edges of domain' ,name='Arial',weight='bold',size=20)
    # plt.xlabel('Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([250,350])
    # ax.set_ylim([0,2])
    
   
    
    
    # ax=fig.add_subplot(2,2,1)
    # plt.plot(theta[k,:,0,0],z)
    # plt.title('Edges of domain' ,name='Arial',weight='bold',size=20)
    # plt.xlabel('Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([290,400])
    # ax.set_ylim([0,14])
#    
#    ax=fig.add_subplot(2,2,2)
#    plt.plot(theta[k,:,0,0],z)
#    plt.title('Center of domain' ,name='Arial',weight='bold',size=20)
#    plt.xlabel('Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    ax.set_xlim([290,400])
#    ax.set_ylim([0,14])
#    
#    ax=fig.add_subplot(2,2,3)
#    plt.plot(thetaV[k,:,0,0],z)
#    plt.title('Edges of domain' ,name='Arial',weight='bold',size=20)
#    plt.xlabel('Virtual Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    ax.set_xlim([290,400])
#    ax.set_ylim([0,14])
#    
#    ax=fig.add_subplot(2,2,4)
#    plt.plot(thetaV[k,:,0,0],z)
#    plt.title('Center of domain' ,name='Arial',weight='bold',size=20)
#    plt.xlabel('Virtual Potential temperature (K)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    ax.set_xlim([290,400])
#    ax.set_ylim([0,14])
    
    
    # ax=fig.add_subplot(2,1,2)
    # print (xparcel[k,1500]/1000.0,zparcel[k,1500]/1000.0)
    # plt.pcolormesh(xm,zm,w[k,:-1,0,:],cmap='seismic',vmin=-0.1, vmax=0.1)
    # plt.colorbar(label='Vertical velocity (m/s)')
    # xp=[]
    # zp=[]
    # for kp in range(0,6000,4):
    #     xp.append(xparcel[k,kp]/1000.0)
    #     zp.append(zparcel[k,kp]/1000.0)
    # plt.scatter(xp,zp)
    # #plt.title(time2[k],name='Arial',weight='bold',size=20)
    # #plt.scatter([xparcel[k,1497]/1000.0,xparcel[k,1500]/1000.0,xparcel[k,1503]/1000.0],[zparcel[k,1497]/1000.0,zparcel[k,1500]/1000.0,zparcel[k,1503]/1000.0])
    # #plt.scatter([xparcel[k,897]/1000.0,xparcel[k,900]/1000.0,xparcel[k,903]/1000.0],[zparcel[k,897]/1000.0,zparcel[k,900]/1000.0,zparcel[k,903]/1000.0])
    # #plt.scatter([xparcel[k,2097]/1000.0,xparcel[k,2100]/1000.0,xparcel[k,2103]/1000.0],[zparcel[k,2097]/1000.0,zparcel[k,2100]/1000.0,zparcel[k,2103]/1000.0])
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,6])

    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,dwdt[k,:,0,:],np.arange(-0.0000075,0.0000076,0.0000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Calculated local change in W')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])  
    
    
    # ax=fig.add_subplot(1,1,1)
    # plt.contourf(xm,zm,(PGFpertwPi+Bw+hadvw+vadvw+whturb+wvturb+wrdamp)[k,:-1,0,:],np.arange(-0.0000075,0.0000076,0.0000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Local change in W from output')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(1,1,1)
    # plt.contourf(xm,zm,dwdtcm1[k,:-1,0,:],cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Local change in W from output')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    
    
#    ax=fig.add_subplot(2,1,2)
#    plt.contourf(xm,zm,advtotal[k,:,0,:],np.arange(-0.0000075,0.0000076,0.0000001),cmap='seismic')
#    #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
#    plt.colorbar(label='Total advection of W')
#    #plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    #ax.set_ylim([0,14])
#    ax.set_xlim([-1000,1000])
#    ax.set_ylim([0,7])
    
    
#    ax=fig.add_subplot(2,1,1)
#    plt.contourf(xm,zm,advwx[k,:,0,:],np.arange(-0.0000075,0.0000076,0.0000001),cmap='seismic')
#    #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
#    plt.colorbar(label='Advection of W by U wind')
#    #plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    #ax.set_ylim([0,14])
#    ax.set_xlim([-1000,1000])
#    ax.set_ylim([0,7])
#    
#    
#    ax=fig.add_subplot(2,1,1)
#    plt.contourf(xm,zm,advwz[k,:,0,:],np.arange(-0.0000075,0.0000076,0.0000001),cmap='seismic')
#    #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
#    plt.colorbar(label='Advection of W by W wind')
#    #plt.title(time1[k],name='Arial',weight='bold',size=20)
#    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
#    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
#    #ax.set_ylim([0,14])
#    ax.set_xlim([-1000,1000])
#    ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,PGFw[k,:,0,:],np.arange(-9.91,-9.765,0.005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Pressure gradient force in W')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,PGFpertw[k,:,0,:],np.arange(-0.55,0.60,0.05),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Perturbation pressure gradient force in W')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,PGFpertwPi[k,:-1,0,:],np.arange(-0.56,0.56,0.005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Pi PFG perturbation')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    ax=fig.add_subplot(2,1,2)
    plt.contourf(xm,zm,Bw[k,:-1,0,:],np.arange(-0.56,0.56,0.005),cmap='seismic')
    #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    plt.colorbar(label='Buoyancy')
    #plt.title(time1[k],name='Arial',weight='bold',size=20)
    plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    #ax.set_ylim([0,14])
    ax.set_xlim([-1000,1000])
    ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,(PGFpertwPi + Bw)[k,:-1,0,:],np.arange(-0.00035,0.00036,0.00002),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Buoyancy plus PGFPipert')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,hadvw[k,:-1,0,:],np.arange(-0.0000035,0.0000036,0.0000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Horizontal advection of w')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,vadvw[k,:-1,0,:],np.arange(-0.0000035,0.0000036,0.0000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Vertical advection of w')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,(vadvw+hadvw)[k,:-1,0,:],np.arange(-0.0000035,0.0000036,0.0000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Vertical advection of w')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-2000,2000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,hidiffw[k,:-1,0,:],np.arange(-0.000000015,0.000000016,0.000000001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='horizontal dissipiation')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,vidiffw[k,:-1,0,:],np.arange(-0.000000000082,0.000000000087,0.000000000005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='vertical dissipiation')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,dudt[k,:,0,:],np.arange(-0.0013,0.0013,0.0001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Calculated local change in U')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7]) 

    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,(-PGFpertuPi+hadvu+vadvu+fcoru+urdamp+upblten)[k,:,0,:-1],np.arange(-0.0013,0.0013,0.0001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Local change in U from output')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,PGFpertuPi[k,:,0,:-1],np.arange(-0.56,0.56,0.005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Pi PFG perturbation for u eq')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,hadvu[k,:,0,:-1],np.arange(-0.00035,0.00036,0.00001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Horizontal advection of u')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-3000,3000])
    # ax.set_ylim([0,7])
    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,vadvu[k,:,0,:-1],np.arange(-0.00035,0.00036,0.00001),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Vertical advection of u')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-3000,3000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,1)
    # plt.contourf(xm,zm,fcoru[k,:,0,:-1],np.arange(-0.00052,0.002,0.0001),cmap='CMRmap')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Coriolis force')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-3000,3000])
    # ax.set_ylim([0,7])
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,upblten[k,:,0,:-1],np.arange(-0.008,0.0085,0.0005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='PBL tendency')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-1000,1000])
    # ax.set_ylim([0,7]) 
    
    
    # ax=fig.add_subplot(2,1,2)
    # plt.contourf(xm,zm,urdamp[k,:,0,:-1],np.arange(-0.0005,0.00055,0.00005),cmap='seismic')
    # #plt.contourf(xm,zh[0,:,0,:]/1000.0,T[k,:,0,:],np.arange(205,315,2),cmap='CMRmap')
    # plt.colorbar(label='Rayleigh damping')
    # #plt.title(time1[k],name='Arial',weight='bold',size=20)
    # plt.xlabel('X Domain (km)',name='Arial',weight='bold',size=16,style='italic')
    # plt.ylabel('Height (km)',name='Arial',weight='bold',size=16,style='italic')
    # #ax.set_ylim([0,14])
    # ax.set_xlim([-3000,3000])
    # ax.set_ylim([0,7])
    
    

    
    
    plt.subplots_adjust(bottom=0.07, top=0.93, hspace=0.2)
    plt.pause(0.5)
    nameoffigure = time2[k] #+ "0"
    string_in_string = "{}".format(nameoffigure)
    plt.savefig("/home/owner/Documents/LLJConvection/cm1model/figures/"+string_in_string)
    plt.close()
    
    
    
#plt.show()
  
#%%   
 















 

