import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def liklihood(x, mu, covar):
    a = np.dot((x-mu).T, np.linalg.inv(covar))
    b = np.dot(a, (x-mu))

    power = -0.5*b

    px = (1/(2*(np.pi)*np.linalg.det(covar)))*math.e**power

    return px

def accuracy_rate(test, predictions):
	correct = 0
	for i in range(len(test)):
		if test[i] == predictions[i]:
			correct += 1
	return (correct / float(len(test))) * 100.0


df1 = pd.read_csv('nl_class1.csv')
df2 = pd.read_csv('nl_class2.csv')
df3 = pd.read_csv('nl_class3.csv')

df1_train, df1_test = df1[:250], df1[250:500]
df2_train, df2_test = df2[:250], df2[250:500]
df3_train, df3_test = df3[:250], df3[250:500]

df1_test.insert(2, "Class", 0, True)
df2_test.insert(2, "Class", 1, True)
df3_test.insert(2, "Class", 2, True)

df_test = pd.concat([df1_test, df2_test, df3_test])
df_test = df_test.sample(frac=1).reset_index().drop('index', axis=1)

mu1 = df1_train.mean()
mu2 = df2_train.mean()
mu3 = df3_train.mean()

vr1 = df1_train.var()
vr2 = df2_train.var()
vr3 = df3_train.var()


dfc1 = df_test.drop('Class', axis=1)
true = df_test['Class'].to_list()
ans_nonlinear = []
# C1 Classifier------------------------------------------------>
avg_var = (vr1[0] + vr1[1] + vr2[0] + vr2[1] + vr3[0] + vr3[1])/6

c1 = np.array([[avg_var, 0], [0, avg_var]])

pred1 = []
for i in range(len(df_test)):
    x = dfc1.iloc[i, :].to_numpy()
    if(liklihood(x, mu1, c1) > max(liklihood(x, mu2, c1), liklihood(x, mu3, c1))):
        pred1.append(0)
    if(liklihood(x, mu2, c1) > max(liklihood(x, mu1, c1), liklihood(x, mu3, c1))):
        pred1.append(1)
    if(liklihood(x, mu3, c1) > max(liklihood(x, mu1, c1), liklihood(x, mu2, c1))):
        pred1.append(2)

print("C1 Classfier: ", accuracy_rate(true, pred1))

acc1 = accuracy_score(true, pred1)
prec1 = precision_score(true, pred1, average='weighted')
f1 = f1_score(true, pred1, average='weighted')
recall1 = recall_score(true, pred1, average='weighted')
ans = [acc1, prec1, f1, recall1]
ans_nonlinear.append(ans)
x = list(dfc1['Column1'])
y = list(dfc1['Column2'])

x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
for i in range(len(x)):
    if pred1[i]==0:
        x1.append(x[i])
        y1.append(y[i])
    if pred1[i]==1:
        x2.append(x[i])
        y2.append(y[i])
    if pred1[i]==2:
        x3.append(x[i])
        y3.append(y[i])

plt.scatter(x1, y1, label = 'Class2')
plt.scatter(x2, y2, label = 'Class1')
plt.scatter(x3, y3, label = 'Class3')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("C1 Classifier")
plt.legend()
plt.show()

# C2 Classifier--------------------------------------------------------->

cov1 = df1_train.cov().to_numpy()
cov2 = df2_train.cov().to_numpy()
cov3 = df3_train.cov().to_numpy()

c2 = (cov1 + cov2 + cov3)/3
pred2 = []
for i in range(len(df_test)):
    x = dfc1.iloc[i, :].to_numpy()
    if(liklihood(x, mu1, c2) > max(liklihood(x, mu2, c2), liklihood(x, mu3, c2))):
        pred2.append(0)
    if(liklihood(x, mu2, c2) > max(liklihood(x, mu1, c2), liklihood(x, mu3, c2))):
        pred2.append(1)
    if(liklihood(x, mu3, c2) > max(liklihood(x, mu1, c2), liklihood(x, mu2, c2))):
        pred2.append(2)

print("C2 Classifier: ", accuracy_rate(true, pred2))
acc1 = accuracy_score(true, pred2)
prec1 = precision_score(true, pred2, average='weighted')
f1 = f1_score(true, pred2, average='weighted')
recall1 = recall_score(true, pred2, average='weighted')
ans = [acc1, prec1, f1, recall1]
ans_nonlinear.append(ans)
x = list(dfc1['Column1'])
y = list(dfc1['Column2'])

x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
for i in range(len(x)):
    if pred2[i]==0:
        x1.append(x[i])
        y1.append(y[i])
    if pred2[i]==1:
        x2.append(x[i])
        y2.append(y[i])
    if pred2[i]==2:
        x3.append(x[i])
        y3.append(y[i])

plt.scatter(x1, y1, label = 'Class2')
plt.scatter(x2, y2, label = 'Class1')
plt.scatter(x3, y3, label = 'Class3')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("C2 Classifier")
plt.legend()
plt.show()

# C3 Classifier---------------------------------------------------------------------------------->

cc1 = np.array([[vr1[0], 0], [0, vr1[1]]])
cc2 = np.array([[vr2[0], 0], [0, vr2[1]]])
cc3 = np.array([[vr3[0], 0], [0, vr3[1]]])

pred3 = []
for i in range(len(df_test)):
    x = dfc1.iloc[i, :].to_numpy()
    if(liklihood(x, mu1, cc1) > max(liklihood(x, mu2, cc2), liklihood(x, mu3, cc3))):
        pred3.append(0)
    if(liklihood(x, mu2, cc2) > max(liklihood(x, mu1, cc1), liklihood(x, mu3, cc3))):
        pred3.append(1)
    if(liklihood(x, mu3, cc3) > max(liklihood(x, mu1, cc1), liklihood(x, mu2, cc2))):
        pred3.append(2)

print("C3 Classifier: ", accuracy_rate(true, pred3))
acc1 = accuracy_score(true, pred3)
prec1 = precision_score(true, pred3, average='weighted')
f1 = f1_score(true, pred3, average='weighted')
recall1 = recall_score(true, pred3, average='weighted')
ans = [acc1, prec1, f1, recall1]
ans_nonlinear.append(ans)
x = list(dfc1['Column1'])
y = list(dfc1['Column2'])

x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
for i in range(len(x)):
    if pred3[i]==0:
        x1.append(x[i])
        y1.append(y[i])
    if pred3[i]==1:
        x2.append(x[i])
        y2.append(y[i])
    if pred3[i]==2:
        x3.append(x[i])
        y3.append(y[i])

plt.scatter(x1, y1, label = 'Class2')
plt.scatter(x2, y2, label = 'Class1')
plt.scatter(x3, y3, label = 'Class3')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("C3 Classifier")
plt.legend()
plt.show()

# C4 Classifier------------------------------------------------------------------------------------->

c_1 = df1_train.cov().to_numpy()
c_2 = df2_train.cov().to_numpy()
c_3 = df3_train.cov().to_numpy()

pred4 = []
for i in range(len(df_test)):
    x = dfc1.iloc[i, :].to_numpy()
    if(liklihood(x, mu1, c_1) > max(liklihood(x, mu2, c_2), liklihood(x, mu3, c_3))):
        pred4.append(0)
    if(liklihood(x, mu2, c_2) > max(liklihood(x, mu1, c_1), liklihood(x, mu3, c_3))):
        pred4.append(1)
    if(liklihood(x, mu3, c_3) > max(liklihood(x, mu1, c_1), liklihood(x, mu2, c_2))):
        pred4.append(2)

print("C4 Classifier: ", accuracy_rate(true, pred4))

acc1 = accuracy_score(true, pred4)
prec1 = precision_score(true, pred4, average='weighted')
f1 = f1_score(true, pred4, average='weighted')
recall1 = recall_score(true, pred4, average='weighted')
ans = [acc1, prec1, f1, recall1]
ans_nonlinear.append(ans)

x = list(dfc1['Column1'])
y = list(dfc1['Column2'])

x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
for i in range(len(x)):
    if pred4[i]==0:
        x1.append(x[i])
        y1.append(y[i])
    if pred4[i]==1:
        x2.append(x[i])
        y2.append(y[i])
    if pred4[i]==2:
        x3.append(x[i])
        y3.append(y[i])

plt.scatter(x1, y1, label = 'Class2')
plt.scatter(x2, y2, label = 'Class1')
plt.scatter(x3, y3, label = 'Class3')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("C4 Classifier")
plt.legend()
plt.show()

ans_nonlinear=pd.DataFrame(ans_nonlinear,columns=["Acc.","Prec.","Recall","F-score"])
ans_nonlinear['id']=["C1","C2",'C3','c4']
ans_nonlinear.set_index('id', inplace=True )
print(ans_nonlinear)