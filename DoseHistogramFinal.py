import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from os import listdir
from os.path import isfile, join


mypath = 'DataOut'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
Proton_energy='1_2MeV'
Folder_in_plots='1.2MeV dose'
"""
1.2mev =12,13
1.4.mev=...
1.8 mev same
8mev =8,9
"""
lol1=12;lol2=13
filter_object = filter(lambda a: Proton_energy in a, onlyfiles);filter_object=list(filter_object)
"""
filter_object_exclude = filter(lambda a: '1_8MeV' in a,filter_object)
filter_object_exclude=list(filter_object_exclude)
for str in filter_object_exclude:
    filter_object.remove(str)
"""
filter_object_exi = filter(lambda a: 'Exi' in a, filter_object)
filter_object_ion=filter(lambda a: 'Ion' in a, filter_object)
filter_object_dose=list(filter(lambda a: 'Dose' in a, filter_object))
list_ion_exi=list(filter_object_exi)+list(filter_object_ion)

list=filter_object_dose

mu=np.zeros(len(list))
sigma=np.zeros(len(list))
a='Nuc'
for i,str in enumerate(list):
    temp=open(mypath+'/'+str,'r');temp=temp.readlines()
    temp=np.array(temp,dtype=np.float32)
    mu[i],sigma[i]=norm.fit(temp)
    print(str,mu[i],sigma[i])

str=list[lol1]
temp=open(mypath+'/'+str,'r');temp=temp.readlines()
temp=np.array(temp,dtype=np.float32)
str2=list[lol2]
temp2=open(mypath+'/'+str2,'r');temp2=temp2.readlines()
temp2=np.array(temp2,dtype=np.float32)
print(str,str2)

print(mu[lol1],mu[lol2])
print(np.abs(mu[lol1]-mu[lol2]),sigma[lol1]+sigma[lol2])

plt.figure(figsize=(10, 4.4))
plt.title('Dose deliverd to cells and nucleus',fontsize=13)
plt.hist(temp2,histtype=u'step',label='Nuclear dose')
plt.hist(temp,histtype=u'step',label='Cellular dose')
plt.grid()
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel('Dose [Gy]',fontsize=13)
plt.ylabel('Number of cells',fontsize=13)
plt.legend()

plt.savefig("Plots/"+Folder_in_plots+"/"+Proton_energy+"3gy1000cellsdose.png",bbox_inches='tight')
