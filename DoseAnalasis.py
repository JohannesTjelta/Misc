import numpy as np
from fwhm import fwhm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

DoseArrayNuc8 = ['Dose1.0Gy1000cellsNuc8MeV','Dose2.0Gy2000cellsNuc8MeV','Dose3.0Gy2000cellsNuc8MeV'  # 'Dose2.0Gy1000cellsNuc8MeV',
                ,'Dose5.0Gy100cellsNuc8MeV','Dose8.0Gy500cellsNuc8MeV','Dose10.0Gy1000cellsNuc8MeV']
DoseArrayNuc1_8 = ['Dose1.0Gy1000cellsNuc1_8MeV','Dose2.0Gy1000cellsNuc1_8MeV','Dose3.0Gy1000cellsNuc1_8MeV'
                ,'Dose5.0Gy1000cellsNuc1_8MeV','Dose8.0Gy3000cellsNuc1_8MeV','Dose10.0Gy1000cellsNuc1_8MeV']
DoseArrayNuc1_4 = ['Dose1.0Gy1000cellsNuc1_4MeV','Dose2.0Gy1000cellsNuc1_4MeV','Dose3.0Gy1000cellsNuc1_4MeV'
                ,'Dose5.0Gy1000cellsNuc1_4MeV','Dose8.0Gy1000cellsNuc1_4MeV','Dose10.0Gy1000cellsNuc1_4MeV']
DoseArrayNuc1_2 = ['Dose1.0Gy1000cellsNuc1_2MeV','Dose2.0Gy1000cellsNuc1_2MeV','Dose3.0Gy1000cellsNuc1_2MeV'
                ,'Dose5.0Gy1000cellsNuc1_2MeV','Dose8.0Gy1000cellsNuc1_2MeV','Dose10.0Gy1000cellsNuc1_2MeV']


DoseArray8 = ['Dose1.0Gy1000cells8MeV','Dose2.0Gy2000cells8MeV','Dose3.0Gy2000cells8MeV'  # 'Dose2.0Gy1000cells8MeV',
                ,'Dose5.0Gy100cells8MeV','Dose8.0Gy500cells8MeV','Dose10.0Gy1000cells8MeV']
DoseArray1_8 = ['Dose1.0Gy1000cells1_8MeV','Dose2.0Gy1000cells1_8MeV','Dose3.0Gy1000cells1_8MeV'
                ,'Dose5.0Gy1000cells1_8MeV','Dose8.0Gy1000cells1_8MeV','Dose10.0Gy1000cells1_8MeV']
DoseArray1_4 = ['Dose1.0Gy1000cells1_4MeV','Dose2.0Gy1000cells1_4MeV','Dose3.0Gy1000cells1_4MeV'
                ,'Dose5.0Gy1000cells1_4MeV','Dose8.0Gy1000cells1_4MeV','Dose10.0Gy1000cells1_4MeV']
DoseArray1_2 = ['Dose1.0Gy1000cells1_2MeV','Dose2.0Gy1000cells1_2MeV','Dose3.0Gy1000cells1_2MeV'
                ,'Dose5.0Gy1000cells1_2MeV','Dose8.0Gy1000cells1_2MeV','Dose10.0Gy1000cells1_2MeV']


def DoseAnalasis(IonArrayNuc,IonArrayCell,string,plotname):
    path = 'DataOut'
    document='.txt'
    sigma=np.zeros(len(IonArrayNuc))
    mu=np.zeros(len(IonArrayNuc))
    fig= plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for i,str in enumerate(IonArrayNuc):
        ionArray =np.loadtxt(path+'/'+str+document)
        hist,bins= np.histogram(ionArray,bins=30,density=True)
        bins_middle= np.zeros(len(hist))
        for ii in range(len(hist)):
            bins_middle[ii]=bins[ii]-np.abs(bins[ii+1]-bins[ii])
        axs[1].plot(bins_middle,hist)
        mu[i],sigma[i]=norm.fit(ionArray)

    sigmaC=np.zeros(len(IonArrayNuc))
    muC=np.zeros(len(IonArrayNuc))
    for i,str in enumerate(IonArrayCell):
        ionArray =np.loadtxt(path+'/'+str+document)
        histC,binsC= np.histogram(ionArray,bins=30,density=True)
        bins_middleC= np.zeros(len(histC))
        for ii in range(len(histC)):
            bins_middleC[ii]=binsC[ii]-np.abs(binsC[ii+1]-binsC[ii])
        axs[0].plot(bins_middleC,histC)
        muC[i],sigmaC[i]=norm.fit(ionArray)
#    np.savetxt('Program/DataDump/'+string+'mu.csv',mu,delimiter=',')
#    pd.DataFrame(mu).to_csv('Program/DataDump/'+string+'mu.csv',header='complex')
#    np.savetxt('Program/DataDump/'+string+'sigma.csv',sigma,delimiter=',')

    axs[0].grid()
    axs[1].grid()
    plt.xlabel('Dose [Gy]',fontsize=13)
    axs[0].set_ylabel('Cells')
    axs[1].set_ylabel('Nuclei')
    fig.text(0.06, 0.5, 'Ammount of cells [Normalized]', fontsize=13,ha='center'
            , va='center', rotation='vertical')

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    fig.suptitle('Dose distrebution for {} protons'.format(string),fontsize=13)
    axs[0].legend(('1Gy    $\sigma$={:1.3f}'.format(sigmaC[0]),'2Gy    $\sigma$={:1.3f}'.format(sigmaC[1]),
                '3Gy    $\sigma$={:1.3f}'.format(sigmaC[2]),'5Gy    $\sigma$={:1.3f}'.format(sigmaC[3]),
                '8Gy    $\sigma$={:1.3f}'.format(sigmaC[4]),'10Gy  $\sigma$={:1.3f}'.format(sigmaC[5])))
    axs[1].legend(('1Gy    $\sigma$={:1.3f}'.format(sigma[0]),'2Gy    $\sigma$={:1.3f}'.format(sigma[1]),
                '3Gy    $\sigma$={:1.3f}'.format(sigma[2]),'5Gy    $\sigma$={:1.3f}'.format(sigma[3]),
                '8Gy    $\sigma$={:1.3f}'.format(sigma[4]),'10Gy  $\sigma$={:1.3f}'.format(sigma[5])))
    plt.savefig('Plots/DoseAnalasis/DoseDistNuc'+plotname,bbox_inches='tight')
    #plt.show()
    plt.close()
    return sigma,mu,sigmaC,muC


def IonAnalasis(IonArrayNuc,string,plotname):
    path = 'DataOut'
    document='.txt'
    sigma=np.zeros(len(IonArrayNuc))
    mu=np.zeros(len(IonArrayNuc))
    plt.figure(figsize=(10, 4.4))
    for i,str in enumerate(IonArrayNuc):
        ionArray =np.loadtxt(path+'/'+str+document)
        histIon,binsIon= np.histogram(ionArray,bins=30,density=True)
        bins_middleIon= np.zeros(len(histIon))
        for ii in range(len(histIon)):
            bins_middleIon[ii]=binsIon[ii]-np.abs(binsIon[ii+1]-binsIon[ii])

        mu[i],sigma[i]=norm.fit(ionArray)
        plt.hist(ionArray,histtype=u'step',bins=30,density=True)


    plt.grid()
    plt.xlabel('Dose [Gy]',fontsize=13)
    plt.ylabel('Ammount of cells [Normalized]',fontsize=13)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.title('Event distrebution for nuclei for {} protons'.format(string),fontsize=13)
    plt.legend(('1.2MeV    $\sigma$={:1.3f}'.format(sigma[0]),'1.5MeV    $\sigma$={:1.3f}'.format(sigma[1]),
                '1.8MeV    $\sigma$={:1.3f}'.format(sigma[2]),'8.7MeV    $\sigma$={:1.3f}'.format(sigma[3])))
    #plt.savefig('Plots/DoseAnalasis/DoseDistNuc'+plotname,bbox_inches='tight')
    #return sigma,mu


if __name__=='__main__':

    sigma1_2,mu1_2,sigmacell1_2,mucell1_2=DoseAnalasis(DoseArrayNuc1_2,DoseArray1_2,' 1.2MeV','1_1MeV.png')
    sigma1_4,mu1_4,sigmacell1_4,mucell1_4=DoseAnalasis(DoseArrayNuc1_4,DoseArray1_4,' 1.5MeV','1_5MeV.png')
    sigma1_8,mu1_8,sigmacell1_8,mucell1_8=DoseAnalasis(DoseArrayNuc1_8,DoseArray1_8,' 1.8MeV','1_8MeV.png')
    sigma8,mu8,sigmacell8,mucell8=DoseAnalasis(DoseArrayNuc8,DoseArray8,' 8.7MeV','8_7MeV.png')
    temp1=np.concatenate((sigma1_2,sigma1_4),axis=0)
    temp2=np.concatenate((sigma1_8,sigma8),axis=0)
    sigma=np.concatenate((temp1,temp2),axis=0)
    temp1=np.concatenate((mu1_2,mu1_4),axis=0)
    temp2=np.concatenate((mu1_8,mu8),axis=0)
    mu=np.concatenate((temp1,temp2),axis=0)
    Energy=np.array((1.2,1.2,1.2,1.2,1.2,1.2,1.5,1.5,1.5,1.5,1.5,1.5,
                    1.8,1.8,1.8,1.8,1.8,1.8,8.7,8.7,8.7,8.7,8.7,8.7))
    print(Energy)
    cellCulture=  {'Energy': Energy,
                    'sigma': sigma,
                    'mu': np.sqrt(mu)
                    }
    from sklearn.model_selection import train_test_split
    df=pd.DataFrame(cellCulture,columns=['Energy','sigma','mu'])
    pd.DataFrame(df).to_csv('Program/DataDump/'+'samemss'+'.csv',header=['Energy','sigma','mu'])
    X=df[['Energy','mu']]
    y=df['sigma']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9)
    regr = linear_model.LinearRegression()
    totz=regr.fit(X, y)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)



    """
    sigmacell1_2,mucell1_2=DoseAnalasis(DoseArray1_2,'cells, 1.2MeV','cell1_2MeV.png')
    sigmacell1_4,mucell1_4=DoseAnalasis(DoseArray1_4,'cells, 1.5MeV','cell1_5MeV.png')
    sigmacell1_8,mucell1_8=DoseAnalasis(DoseArray1_8,'cells, 1.8MeV','cell1_8MeV.png')
    sigmacell8,mucell8=DoseAnalasis(DoseArray8,'cells, 8.7MeV','cell8_7MeV.png')
    """
    mu_tot_1_2=(mucell1_2-mu1_2)
    mu_tot_1_4=(mucell1_4-mu1_4)
    mu_tot_1_8=(mucell1_8-mu1_8)
    mu_tot_8=(mucell8-mu8)
    sigma_tot_1_2=abs(sigma1_2+sigmacell1_2)
    sigma_tot_1_4=abs(sigma1_2+sigmacell1_2)
    sigma_tot_1_8=abs(sigma1_2+sigmacell1_2)
    sigma_tot_8=abs(sigma1_2+sigmacell1_2)
    plt.close()

    meandose=np.array((1,2,3,5,8,10))
    fig= plt.figure(figsize=(10, 4.4))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.plot(meandose,mu_tot_1_2,'.',color='black')
    plt.plot(meandose,mu_tot_1_4,'.',color='black')
    plt.plot(meandose,mu_tot_1_8,'.',color='black')
    plt.plot(meandose,mu_tot_8,'.',color='black')
    plt.plot(meandose,mu_tot_1_8,'-',label='1.8MeV',color='red')
    plt.plot(meandose,mu_tot_1_4,'-',label='1.4MeV',color='blue')
    plt.plot(meandose,mu_tot_1_2,'-',label='1.2MeV',color='green')
    plt.plot(meandose,mu_tot_8,'-',label='8.7MeV',color='yellow')
    plt.legend()
    plt.grid()
    plt.xlabel('Expected dose [Gy]',fontsize=13)
    plt.ylabel('mean differance [Gy]',fontsize=13)
    def complex(mu,T):
        return 0.27*T**(-1/2)*mu**(1./2)-0.17*T**(-1/2)
        #return 0.094*(1/np.sqrt(T))*np.sqrt(mu)

    def simple(mu,T):
        return 0.13*(1/np.sqrt(T))*np.sqrt(mu)-0.04
    t=np.array((1.2,1.2,1.2,1.2,1.2,1.2))
    t14=np.array((1.4,1.4,1.4,1.4,1.4,1.4))
    t18=np.ones(6)*1.8
    t8=np.ones(6)*8.7
    fig= plt.figure(figsize=(10, 4.4))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.plot(mu1_2,sigma1_2,'.',label='1.2 MeV')
    plt.plot(mu1_4,sigma1_4,'.',label='1.4 MeV')
    plt.plot(mu1_8,sigma1_4,'.',label='1.8 MeV')
    plt.plot(mu8,sigma8,'.',label='8.7 MeV')
    plt.plot(mu1_2,simple(mu1_2,t),label='Model 1.2Mev')
    plt.plot(mu1_4,simple(mu1_4,t14),label='Model 1.4Mev')
    plt.plot(mu1_8,simple(mu1_8,t18),label='Model 1.8Mev')
    plt.plot(mu8,simple(mu8,t8),label='Model 8.7Mev')
    plt.grid()
    plt.legend()
    plt.xlabel('Mean dose [Gy]',fontsize=13)
    plt.ylabel('Standard deviation [Gy]',fontsize=13)
    plt.savefig('Plots/DoseDistNucm.png',bbox_inches='tight')


    #slope, intercept, r_value, p_value, std_err
    value1_2 = stats.linregress(np.sqrt(mu1_2),sigma1_2)
    log_regress_1_2=np.polyfit(np.log(mu1_2), sigma1_2, 1)
    value1_4 = stats.linregress(np.sqrt(mu1_4),sigma1_4)
    value1_8 = stats.linregress(np.sqrt(mu1_8),sigma1_8)
    value8 = stats.linregress(np.sqrt(mu8),sigma8)
    print(value8)
    value= stats.linregress(1./np.array((8.7,1.8,1.4,1.2))**(1/2),(value8[0],value1_8[0],value1_4[0],value1_2[0]))
    print(value)
    plt.savefig('Plots/Udesignert/stdandstuff.png',bbox_inches='tight')
    plt.figure()
    plt.plot((8.7,1.8,1.4,1.2),(value8[0],value1_8[0],value1_4[0],value1_2[0]),'.')
    plt.xlabel('Energy [MeV]')
    plt.ylabel('Std')
    plt.savefig('temp/')

    fig= plt.figure(figsize=(10, 4.4))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.plot(np.linspace(0.5,100,1000),complex(1,np.linspace(0.5,100,1000)),label='1Gy C')
    plt.plot(np.linspace(0.5,100,1000),complex(3,np.linspace(0.5,100,1000)),label='3Gy C')
    plt.plot(np.linspace(0.5,100,1000),complex(5,np.linspace(0.5,100,1000)),label='5Gy C')
    plt.plot(np.linspace(0.5,100,1000),complex(10,np.linspace(0.5,100,1000)),label='10Gy C')
    plt.plot(np.linspace(0.5,100,1000),simple(1,np.linspace(0.5,100,1000)),'--',label='1Gy S')
    plt.plot(np.linspace(0.5,100,1000),simple(3,np.linspace(0.5,100,1000)),'--',label='3Gy S')
    plt.plot(np.linspace(0.5,100,1000),simple(5,np.linspace(0.5,100,1000)),'--',label='5Gy S')
    plt.plot(np.linspace(0.5,100,1000),simple(10,np.linspace(0.5,100,1000)),'--',label='10Gy S')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [Gy]',fontsize=13)
    plt.xlabel('Beam energy [MeV]', fontsize=13)
    plt.savefig('Plots/Modelsstd.png',bbox_inches='tight')
