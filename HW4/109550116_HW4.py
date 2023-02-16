import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt   
import seaborn as sns

def cross_validation(x_train, y_train, k=5):
    #combine
    length = y_train.shape[0]
    y_train = y_train.reshape((length,1))
    all_train = np.hstack([x_train,y_train])
    fold=None
    part_a=list()
    if length%k==0:
        part_a = np.vsplit(all_train,k)
    else:
        a=int(length/k)
        b=int(length%k)
        c=(a+1)*b
        part_aa = np.vsplit(all_train[:c,:],b)
        part_bb = np.vsplit(all_train[c:,:],k-b)
        part_a = part_aa+part_bb
        #print(len(part_aa))
        #print(part_aa[0].shape)
        #print(len(part_bb))
    #print(len(part_a))

    for i in range(k):
        val = part_a[i]
        train = None
        jud=0
        for j in range(k):
            if j==i:
                continue
            else:
                if jud == 0:
                    train = part_a[j]
                else:
                    train = np.vstack([train,part_a[j]])
                jud+=1

        combine = [train]
        combine.append(val)
        if i == 0:
            fold=[combine]
        else:
            fold.append(combine)

    #print(len(fold[0]))
    #print(fold[0][0].shape)
    
    return fold

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")

g=[0.001,0.0001,0.00001]
c=[100.0,10.0,1.0]
acc_t = np.zeros((3,3))
#print(acc)

fold = cross_validation(x_train,y_train,k=5)
feat = fold[0][0].shape[1]
print(len(fold))
print(len(fold[0]))
print(fold[0][1].shape[0])



#print(feat)
'''
x_tr=fold[3][0][:,:feat-1]
y_tr=fold[3][0][:,feat-1:].flatten()
x_te=fold[3][1][:,:feat-1]
y_te=fold[3][1][:,feat-1:].flatten()
'''
count = len(fold)

best=0
best_c=0
best_g=0
for i in range(len(c)):
    for j in range(len(g)):
        acc=0
        for k in range(count):
            clf = SVC(C=c[i], kernel='rbf', gamma=g[j])
            x_tr=fold[k][0][:,:feat-1]
            y_tr=fold[k][0][:,feat-1:].flatten()
            x_te=fold[k][1][:,:feat-1]
            y_te=fold[k][1][:,feat-1:].flatten()
            clf.fit(x_tr,y_tr)
            y_pred=clf.predict(x_te)
            acc+=accuracy_score(y_pred,y_te)
        avg=acc/count
        acc_t[i][j]=avg
        if(avg>best):
            best=avg
            best_c=i
            best_g=j

df=pd.DataFrame(acc_t,columns=g,index=c)
sns.heatmap(df,annot=True,cmap='RdBu',vmin=0.4,vmax=1.0)
plt.xlabel("Gamma Parameter")
plt.ylabel("C Parameter")
plt.show()
print("best_parameters c=",c[best_c],", g=",g[best_g])

'''
clf = SVC(C=1.0, kernel='rbf', gamma=0.0005)
clf.fit(x_tr,y_tr)
y_pred=clf.predict(x_te)
#print(y_pred)
print(accuracy_score(y_pred,y_te))
'''

#print(x_train)

best_model = SVC(C=c[best_c], kernel='rbf', gamma=g[best_g])
best_model.fit(x_train,y_train)
y_pred = best_model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))

