from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


k=5
x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
#print(len(x_train))
#x_train = x_test
#y_train = y_test
n = len(x_train)
first_xx = list()
first_yy = list()
sec_xx = list()
sec_yy = list()
f_m_x = 0 
f_m_y = 0 
s_m_x = 0 
s_m_y = 0 
#print(y_train[0])
for i in range(n):
    if(y_train[i]==0):
        f_m_x = f_m_x+x_train[i][0]
        f_m_y = f_m_y+x_train[i][1]
        first_xx.append(x_train[i][0])
        first_yy.append(x_train[i][1])
    else:
        s_m_x = s_m_x+x_train[i][0]
        s_m_y = s_m_y+x_train[i][1]
        sec_xx.append(x_train[i][0])
        sec_yy.append(x_train[i][1])

first_xx = np.array(first_xx)
first_yy = np.array(first_yy)
sec_xx = np.array(sec_xx)
sec_yy = np.array(sec_yy)
#print(first_xx.shape)
#print(first_yy.shape)


plt.scatter(first_xx,first_yy,color='b',s=1,zorder=2)
plt.scatter(sec_xx,sec_yy,color='orange',s=1,zorder=2)
#plt.show()


'''part1'''

f_n = len(first_xx)
s_n = len(sec_xx)
f_m_x = f_m_x/f_n
f_m_y = f_m_y/f_n
s_m_x = s_m_x/s_n
s_m_y = s_m_y/s_n
print(f"mean vector of class 1: {f_m_x,f_m_y}", f"\nmean vector of class 2: {s_m_x,s_m_y}\n")

'''part2'''

#class 1
sw_w = np.zeros((2,2))
for i in range(f_n):
    temp = np.zeros(2)
    temp[0] = first_xx[i]-f_m_x
    temp[1] = first_yy[i]-f_m_y
    temp1 = temp.T.reshape(2, 1)
    temp = temp.reshape(1, 2)
    sw_w += np.dot(temp1,temp) 
#class2
for i in range(s_n):
    temp = np.zeros(2)
    temp[0] = sec_xx[i]-s_m_x
    temp[1] = sec_yy[i]-s_m_y
    temp1 = temp.T.reshape(2, 1)
    temp = temp.reshape(1, 2)
    sw_w += np.dot(temp1,temp) 

print(f"Within-class scatter matrix SW: {sw_w}\n")
#print(sw_w)

'''part3'''

temp = np.zeros(2)
temp[0] = f_m_x-s_m_x
temp[1] = f_m_y-s_m_y
temp1 = temp.T.reshape(2, 1)
temp = temp.reshape(1, 2)
sb = np.dot(temp1,temp)
print(f"Between-class scatter matrix SB: {sb}\n")

'''part4'''

j = np.dot(np.linalg.inv(sw_w),sb)
eigva,eigve = np.linalg.eig(j)
#print(eigva)
#print(eigve)
m = np.array([eigve[0][1],eigve[1][1]])
slope = eigve[1][1]/eigve[0][1]
print(f"Fisher’s linear discriminant: {m}\n")

'''part5'''

t_n = len(x_test)

pro_t = np.zeros((t_n,2))
for i in range(t_n):
    pro_tt = np.dot(x_test[i],m)/np.dot(m,m)*m
    pro_t[i] = pro_tt
    #print(pro_tt)
#print (pro_t.shape)

pro_tr = np.zeros((n,2))
for i in range(n):
    pro_ttr = np.dot(x_train[i],m)/np.dot(m,m)*m
    pro_tr[i] = pro_ttr

dtype = [('dis',float),('class',float)]
acc = list()
y_pred = np.zeros(t_n)
for k in range(1,6):
    for j in range(t_n):
        dis_t = list()
        for i in range(n):
            dis = ((pro_tr[i][0]-pro_t[j][0])**2)+((pro_tr[i][1]-pro_t[j][1])**2)
            dis_t.append((dis,y_train[i]))
        temp = np.array(dis_t,dtype=dtype)
        result = np.sort(temp,order=['dis'])
        jud=0
        for i in range(k):
            jud += result[i][1]
        if(jud>(k/2)):
            y_pred[j]=1
        elif (jud<(k/2)):
            y_pred[j]=0
        else:
            y_pred[j]=result[0][1]
        
    #o = y_pred[j]-y_test[j]
    #if (o!=0):
    #plt.scatter(x_test[j][0],x_test[j][1],color='green',s=3,zorder=3)
    acc = accuracy_score(y_test, y_pred)
    print("K=",k,":")
    print(f"Accuracy of test-set {acc}")

'''part6'''

first_pro_x=list()
first_pro_y=list()
sec_pro_x=list()
sec_pro_y=list()

for i in range(f_n):
    before = np.array([first_xx[i],first_yy[i]])
    pro = np.dot(before,m)/np.dot(m,m)*m
    #print(pro)
    #print(pro[1])
    first_pro_x.append(pro[0])
    first_pro_y.append(pro[1])

first_pro_x = np.array(first_pro_x)
first_pro_y = np.array(first_pro_y)

for i in range(s_n):
    before = np.array([sec_xx[i],sec_yy[i]])
    pro = np.dot(before,m)/np.dot(m,m)*m
    #print(pro)
    #print(pro[1])
    sec_pro_x.append(pro[0])
    sec_pro_y.append(pro[1])


sec_pro_x = np.array(sec_pro_x)
sec_pro_y = np.array(sec_pro_y)
s="Projection line w="+str(slope)+" ,b=0"
plt.title(s)
plt.scatter(sec_pro_x,sec_pro_y,color='red',s=3,zorder=3)
plt.scatter(first_pro_x,first_pro_y,color='green',s=3,zorder=3)
plt.plot([first_xx,first_pro_x],[first_yy,first_pro_y],'--',color='darkgray',linewidth=0.3,zorder=1)
plt.plot([sec_xx,sec_pro_x],[sec_yy,sec_pro_y],'--',color='#cbb1de',linewidth=0.3,zorder=1)
plt.axline((0,0),m,linewidth=1)
plt.show()