import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


path = ['Data1_2','Data1_4','Data1_8','Data8']
proton =['5.root','49.root','1.root','19.root']
Name=['1,1MeV','1,3MeV','1,7MeV','8,5MeV']
dz=4000
for i,str in enumerate(path):
    specificProt=uproot.open(str+'/'+proton[i])['microdosimetry']
    z = specificProt['z'].array()
    particle = specificProt['flagParticle'].array()
    process = specificProt['flagProcess'].array()
    energy = specificProt['totalEnergyDeposit'].array()
    z_index=np.where((z>z[0])&(z<z[0]+dz))
    z=z[z_index]
    energy=energy[z_index]

    particle=particle[z_index]
    process=process[z_index]
    z0=z[0];z=z-z0


    n=100
    dist=np.linspace(0,4000,n)
    energy_deposit_bin=np.zeros(n)
    print(z[-1])
    for ii in range(n-1):
            bin=np.where((z>dist[ii])&(z<dist[ii+1]))
            energy_deposit_bin[ii]=sum(energy[bin])
    fig = plt.figure()
    plt.plot(dist[:-1],energy_deposit_bin[:-1],label=Name[i])
    plt.title('Energy deposited by a single proton')
    plt.xlabel('Dist [nm]')
    plt.ylabel('Energy Deposited [eV]')
    plt.grid()
    plt.legend()
    plt.savefig('Plots/SingleEnergyDeposit/EnergyDeposit'+str+'.png')
