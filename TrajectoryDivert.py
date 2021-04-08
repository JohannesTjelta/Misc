import numpy as np
import uproot
import os

def numberOfFiles(path):
    root_files=os.listdir(path)
    if 'Res'in root_files:
        root_files.remove('Res')
    if 'analasis' in root_files:
        root_files.remove('analasis')

    return(len(root_files))

def TrackDiv(Path):
    ammount=0
    tot_diff=np.zeros(2000)
    invalid_protons1_2=np.array([140,1313, 2128, 2129, 7075,
                                13104, 13105, 11661, 16128 ,16129
                                ,7411,2251,2313,986])
    file_of_interest=list(open('Data1_8/Res/FilesOfInterest.txt','r'))
    for i,str in enumerate(file_of_interest):

        file_of_interest[i]=str.replace('\n','')

    for i,str in enumerate(file_of_interest):
        file=uproot.open(path+'/'+str)['microdosimetry']  # import root file as data set
        x = file['x'].array()  # import position from data set to numpy array
        if len(x)<2:
            continue
        elif Path=='Data1_2' and i==invalid_protons1_2[np.where(invalid_protons1_2==i)]:
            print(123)
            continue
        else:
            ammount+=1

        y = file['y'].array()
        z = file['z'].array()
        particle=file['flagParticle'].array()
        z=z-z[0]

        x=x[np.where(z<4000)];x=x-x[0]
        y=y[np.where(z<4000)];y=y-y[0]
        z=z[np.where(z<4000)]
        if np.amax(x)>6000:# or np.amax(y)>6000:

            x[np.where(x>6000)] = 0
            y[np.where(y>6000)] = 0
        if np.amax(y)>6000:# or np.amax(y)>6000:
            print(np.amax(y))
            x[np.where(x>6000)] = 0
            y[np.where(y>6000)] = 0        #print(np.where(z==np.amax(z)))
        particle=particle[np.where(z<4000)]
        #print(np.sqrt((x[0]-x[np.where(x==np.amax(x))])**2+(y[0]-y[np.where(x==np.amax(x))])**2))
        temp1=(x[0]-x[np.where(x==np.amax(x))])**2
        temp2=(y[0]-y[np.where(x==np.amax(x))])**2
        #print(temp1,temp2)
        temp3=np.sqrt(temp1+temp2)
        if len(temp3)>1:
            temp3=temp3[0]

        tot_diff[i]=temp3
        print(sum(tot_diff[tot_diff!=0])/ammount,np.std(tot_diff[tot_diff!=0]),ammount)


path="Data1_8"
#N=numberOfFiles(path)
TrackDiv(path)
