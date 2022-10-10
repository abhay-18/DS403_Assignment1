import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt 

mel_train=list(pd.read_csv("Segment2_MelEnergy.csv", header=None)[0])
mel_test=list(pd.read_csv("Segment3_MelEnergy.csv", header=None)[0])

ste_train=list(pd.read_csv("Segment2_STEnergy.csv", header=None)[0])
ste_test=list(pd.read_csv("Segment3_STEnergy.csv", header=None)[0])

class_train=list(pd.read_csv("Segment2_VAD_GT.csv", header=None)[0])
class_test=list(pd.read_csv("Segment3_VAD_GT.csv", header=None)[0])

total_mel=0
n_mel=0
for i in range(len(mel_train)):
    if class_train[i]==1:
        total_mel+=mel_train[i]
        n_mel+=1

mean_mel=total_mel/n_mel

total_ste=0
n_ste=0
for i in range(len(ste_train)):
    if class_train[i]==1:
        total_ste+=ste_train[i]
        n_ste+=1

mean_ste=total_ste/n_ste

var_mel=0
for i in range(len(mel_train)):
    if class_train[i]==1:
        var_mel+=(mel_train[i]-mean_mel)**2 

var_mel/=(n_mel-1)
sigma_mel=var_mel**0.5 

var_ste=0
for i in range(len(mel_train)):
    if class_train[i]==1:
        var_ste+=(ste_train[i]-mean_ste)**2 

var_ste/=(n_ste-1)
sigma_ste=var_ste**0.5 


def prob(x,m,s):
    p=math.pi 
    e=math.e 
    return (1/(s*((2*p)**0.5)))*(e**((-((x-m)/s)**2)/2))


tp_mel=[]
fp_mel=[]

tp_ste=[]
fp_ste=[]

for p_sr in np.arange(0,3,0.01):
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(mel_test)):
        x=prob(mel_test[i],mean_mel,sigma_mel )
        if x>p_sr:
            if class_test[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if class_test[i]==1:
                fn+=1
            else:
                tn+=1
    tp_mel.append(tp/(tp+fn))
    fp_mel.append(fp/(tn+fp))
    
    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(ste_test)):
        x=prob(ste_test[i],mean_ste,sigma_ste )
        if x>p_sr:
            if class_test[i]==1:
                tp+=1
            else:
                fp+=1
        else:
            if class_test[i]==1:
                fn+=1
            else:
                tn+=1
    
    tp_ste.append(tp/(tp+fn))
    fp_ste.append(fp/(tn+fp))


plt.plot(fp_mel, tp_mel)
plt.plot(fp_ste, tp_ste)
plt.title("ROC Curve")
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend(["MEL Energy","STE Energy"])
plt.show()