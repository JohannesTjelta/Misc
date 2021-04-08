import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from os import listdir
from os.path import isfile, join


mypath = 'DataOut'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
Proton_energy='1_4MeV'
filter_object = filter(lambda a: Proton_energy in a, onlyfiles);filter_object=list(filter_object)
"""
filter_object_exclude = filter(lambda a: '1_8MeV' in a,filter_object)
filter_object_exclude=list(filter_object_exclude)
for str in filter_object_exclude:
    filter_object.remove(str)
"""
filter_object_exi = filter(lambda a: 'Exi' in a, filter_object)
filter_object_ion=filter(lambda a: 'Ion' in a, filter_object)
filter_object_dose=filter(lambda a: 'Dose' in a, filter_object)
list_ion_exi=list(filter_object_exi)+list(filter_object_ion)

list=list_ion_exi

mu=np.zeros(len(list))
sigma=np.zeros(len(list))
a='Nuc'
for i,str in enumerate(list):
    temp=open(mypath+'/'+str,'r');temp=temp.readlines()
    temp=np.array(temp,dtype=np.float32)
    mu[i],sigma[i]=norm.fit(temp)
    print(str,mu[i],sigma[i])

str=list[10];print(str)
temp=open(mypath+'/'+str,'r');temp=temp.readlines()
temp=np.array(temp,dtype=np.float32)
str2=list[11];print(str2)
temp2=open(mypath+'/'+str2,'r');temp2=temp2.readlines()
temp2=np.array(temp2,dtype=np.float32)
str3=list[28];print(str3)
temp3=open(mypath+'/'+str3,'r');temp3=temp3.readlines()
temp3=np.array(temp3,dtype=np.float32)
str4=list[29];print(str4)
temp4=open(mypath+'/'+str4,'r');temp4=temp4.readlines()
temp4=np.array(temp4,dtype=np.float32)

Folder_in_plots='Ion 1.4MeV'
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.hist(temp,label='Excitations Cells',bins=10)
ax.hist(temp3,label='Ionizations Cells',bins=10)
ax.grid()
ax.hist(temp2,label='Excitations Nucleus',bins=10)
ax.hist(temp4,label='Ionizations Nucleus',bins=10)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
ax.legend()
plt.title('Number of particular events in nucleus and cells',fontsize=13)
plt.ylabel('Number of cells',fontsize=13)
plt.xlabel('Number of events',fontsize=13)
plt.savefig("Plots/"+Folder_in_plots+"/1_8mev3gy2000cells.png",bbox_inches='tight')
plt.show()
