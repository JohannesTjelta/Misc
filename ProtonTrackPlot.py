import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


path = ['Data1_2','Data1_4','Data1_8','Data8']
proton =['2.root','52.root','106.root','19.root']
dz=4000
for i,str in enumerate(path):
    specificProt=uproot.open(str+'/'+proton[i])['microdosimetry']
    x = specificProt['x'].array()
    y = specificProt['y'].array()
    z = specificProt['z'].array()
    particle = specificProt['flagParticle'].array()
    process = specificProt['flagProcess'].array()
    z_index=np.where((z>z[0])&(z<z[0]+dz))
    z=z[z_index]
    x=x[z_index]
    y=y[z_index]
    particle=particle[z_index]
    process=process[z_index]
    x0=x[0];x=x-x0
    y0=y[0];y=y-y0
    z0=z[0];z=z-z0

    #print(len(x[np.where((process==12)|(process==22))]),len(x[np.where((process==13)|(process==23))]),len(z))
    print(str)
    x_el=x[np.where(particle==1)]
    y_el=y[np.where(particle==1)]
    z_el=z[np.where(particle==1)]
    x_p=x[np.where(particle==2)]
    y_p=y[np.where(particle==2)]
    z_p=z[np.where(particle==2)]
    print(z_p[0])
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(x_p,y_p,z_p,'red',label='Proton')
    ax.plot3D(x_el,y_el,z_el,',',label='Electron')
    ax.set_xlabel('Length [nm]')
    ax.set_ylabel('Length [nm]')
    ax.set_zlabel('Length [nm]')
    ax.legend()
    ax.view_init(elev=10.,azim=70)
    plt.subplots_adjust(top=1,bottom=0)
    plt.show()
    #plt.savefig('Plots/ProtonTrack/PtrotonTrack'+str+'.png')
