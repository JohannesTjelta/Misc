import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('seaborn-deep')

Dose = ['Dose3.0Gy1001cellsNuc1_8MeV',
        'Dose3.0Gy1002cellsNuc1_8MeV','Dose3.0Gy1003cellsNuc1_8MeV']
path='DataOut/'
format='.txt'
mu=np.zeros(len(Dose))
sigma=np.zeros(len(Dose))
plt.figure(figsize=(10, 4.4))
bins=np.linspace(2.35,3.15,30)
plt.hist([np.loadtxt(path+Dose[0]+format),np.loadtxt(path+Dose[1]+format),
        np.loadtxt(path+Dose[2]+format)],bins,density=True)
for i,str in enumerate(Dose):
    file=np.loadtxt(path+str+format)
    mu[i],sigma[i]=norm.fit(file)
#plt.hist(file,alpha=0.7,bins=30,density=True,label='$\sigma$={:.3f}'.format(sigma[i]))

plt.title('Stress test with 1.8MeV beam at 3.0Gy',fontsize=13)
plt.xlabel('Dose [Gy]',fontsize=13)
plt.ylabel('Ammount',fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.grid()
plt.legend(['Test 1   $\sigma$={:.3f}'.format(sigma[0]),'Test 2   $\sigma$={:.3f}'.format(sigma[1]),
            'Test 3   $\sigma$={:.3f}'.format(sigma[2])],loc='best')
plt.savefig('Plots/Udesignert/StressTest.png',bbox_inches='tight')
plt.show()
