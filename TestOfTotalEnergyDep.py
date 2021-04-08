import uproot
import numpy as np
import matplotlib.pyplot as plt
def deposit(data,name):
    FilesOfInterestPath=data+'/Res/FilesOfInterest.txt'
    text_file=open(FilesOfInterestPath,'r')#.read()
    lines = text_file.readlines()
    protons =[s.replace('\n','') for s in lines]
    dz=4e3
    n=4000
    tot_deposit=np.zeros(n)
    for root in protons:
        file=uproot.open(data+'/'+root)['microdosimetry']
        z=file['z'].array()
        en = file['totalEnergyDeposit'].array()
        z_delta=z[np.where((z>z[0])&(z<(z[0]+dz)))]
        en_delta=en[np.where((z>z[0])&(z<(z[0]+dz)))]
        en_deposit_temp=np.zeros(n)
        z_dz=np.linspace(z_delta[0],z_delta[0]+dz,n)
        for ii in range(len(z_dz)-1):
            k=np.where((z_delta>z_dz[ii])&(z_delta<z_dz[ii+1]))
            en_deposit_temp[ii]=sum(en_delta[k])
        tot_deposit+=en_deposit_temp
    overall_en_dep=tot_deposit/len(protons)
    plt.plot(z_dz[2:-3],overall_en_dep[2:-3],',')
    plt.ylabel('Energy Deposition [eV]')
    plt.xlabel('Distance traveld [nm]')
    plt.grid()
    plt.title('LET [eV/nm] from the proton pool for'+name)
    plt.savefig(data+'/Res/'+name+'.png')
if __name__=="__main__":
    deposit("Data1_2",'1,1MeV')
