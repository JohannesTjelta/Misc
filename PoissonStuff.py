import numpy as np; import matplotlib.pyplot as plt
import os
from scipy.stats import poisson


test=['1_2MeV1000cellss1GyCellx','1_2MeV1000cellss2GyCellx','1_2MeV1000cellss3GyCellx'
      ,'1_2MeV1000cellss5GyCellx','1_2MeV1000cellss8GyCellx','1_2MeV1000cellss10GyCellx']
test=['1_2MeV1001cellss1GyNucx']
mypath = 'filesxy'
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
cells = '1000'
filter_object = filter(lambda a: cells in a, onlyfiles);filter_object=list(filter_object)
filter_object_x = filter(lambda a: 'x.txt' in a, filter_object);filter_object_x=list(filter_object_x)
filter_object = filter(lambda a: '1_2' in a, filter_object_x);filter_object=list(filter_object)
num_hits=np.zeros(len(filter_object_x)*1001)
num_hits=np.reshape(num_hits,(len(filter_object_x),1001))
fig= plt.figure(figsize=(10, 4.4))
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
for i,str in enumerate(test):
    print(str)
    temp=open(mypath+'/'+str+'.txt','r');temp=temp.readlines()
    print(len(temp))
    for ii,list in enumerate(temp):
        x=list.split()
        x=np.array(x,dtype=np.float32)
        num_hits[i,ii]=len(x[x!=0])
    mu=int(np.mean(num_hits[i,:]))
    dist=np.random.poisson(lam=mu,size=1000)
    plt.hist(dist,alpha=0.7,histtype=u'step',bins=20,density=True,color='blue')
    plt.hist(num_hits[i,:],alpha=0.4,bins=20,density=True,color='red')
    print('yo')
plt.grid()
plt.xlabel('Hits',fontsize=13)
plt.ylabel('Frequensy',fontsize=13)

plt.legend(('Simulated data','Poisson Dist'))
plt.show()
