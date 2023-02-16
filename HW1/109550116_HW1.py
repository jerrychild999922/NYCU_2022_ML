import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1/(1+np.exp(-x))

'''part1'''

x_train, x_test, y_train, y_test = np.load('regression_data.npy', allow_pickle=True)
#plt.plot(x_train, y_train, '.')
#plt.show()
#print(x_train.shape)
#print(x_train[1][0])
x_train = x_train.ravel()
y_train = y_train.ravel()
x_test = x_test.ravel()
y_test = y_test.ravel()
m = np.random.uniform(0,1) 
b = 0
lr = 0.1
epochs = 100  
n = float(len(x_train)) 
los=np.zeros(epochs)

for i in range(epochs): 
    prediction = m * x_train + b  
    #print (Y_pred.shape)
    loss = (1/n) * sum(np.square(y_train - prediction))
    los[i]=loss
    #print(loss)
    D_m = (-2/n) * sum(x_train * (y_train - prediction)) 
    D_b = (-2/n) * sum(y_train - prediction)  
    m = m - lr * D_m  
    b = b - lr * D_b 

print ("\nwight for part1 :",m)
print ("intercepts for part1 :",b)

'''test'''

test_n = float(len(x_test))
y_pred = m * x_test + b
test_loss = (1/test_n) * sum(np.square(y_test - y_pred))
print("Error of test for part1 :",test_loss,"\n")

plt.title("learning curve for part1")
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.plot(los)
plt.show()

'''part2'''
x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
#plt.scatter(x_train, np.ones_like(x_train), c=y_train)
#plt.show()
#print(x_train.shape)
x_train = x_train.ravel()
y_train = y_train.ravel()
x_test = x_test.ravel()
y_test = y_test.ravel()
m = np.random.uniform(0,1)
b = 0
lr = 0.2 
epochs = 5000  

n = float(len(x_train))
#print(n)
los1=np.zeros(epochs)
for i in range(epochs): 
    prediction = sigmoid(m * x_train + b ) 
    #print(prediction.shape)
    #print (Y_pred.shape)
    loss = - np.sum(np.multiply(y_train, np.log(prediction)) + np.multiply((1-y_train), np.log(1-prediction)))
    los1[i]=loss
    #print(loss)
    D_m = (-1/n) * sum(x_train * (y_train - prediction))  
    D_b = (-1/n) * sum(y_train - prediction)  
    m = m - lr * D_m  
    b = b - lr * D_b  

print ("wight for part2 :",m)
print ("intercepts for part2 :",b)


'''test'''

test_n = float(len(x_test))
y_pred = sigmoid(m * x_test + b)
test_loss = - np.sum(np.multiply(y_test, np.log(y_pred)) + np.multiply((1-y_test), np.log(1-y_pred)))
print("Error of test for part2 :",test_loss,"\n")

plt.title("learning curve for part2")
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.plot(los1)
plt.show()