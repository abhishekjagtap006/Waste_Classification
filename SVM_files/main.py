import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
# dir="C:\\Users\\Admin\\Documents\\Classification_ML_2\\DATA\\DATASET\\DATASET\\TEST"
# categories=['O','R']
# data=[]
# for category in categories:
#     path=os.path.join(dir,category)
#     label=categories.index(category)//0
#     for img in os.listdir(path):
#         imgpath=os.path.join(path,img)
#         pet_img=cv2.imread(imgpath,0)
#         try:
#             pet_img=cv2.resize(pet_img,(50,50))
#             image=np.array(pet_img).flatten()
#             data.append([image,label])
#         except Exception as e:
#             pass
#
#
# print(len(data))
#print(data)
# pick_in=open('datas.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()
pick_in=open('datas.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()
# #print(data)
#random.seed( 20 )
#print(data)

random.shuffle(data)
features=[]
labels=[]
for feature,label in data:
    features.append(feature)
    labels.append(label)
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.20)
# model = SVC(C=1,kernel='poly',gamma='auto')
# model.fit(xtrain,ytrain)
# pick_in=open('model.sav','wb')
# pickle.dump(model ,pick_in)
# pick_in.close()
# # model = SVC(C=1,kernel='linear',gamma='auto')
# # model.fit(xtrain,ytrain)
# # pick_in=open('model2.sav','rb')
# # model=pickle.load(pick_in)
# # pick_in.close()
pick=open('model.sav','rb')
model=pickle.load(pick)
pick.close()
# print(model)
prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)
# categories=['O','R']
print('Accuracy: ',accuracy)
#print('prediction: ',prediction[0])
if(prediction[0]==0):
    print("prediction: Recycalble Waste")
else:
    print("prediction: Organic Waste")

mypet=xtest[0].reshape(50,50)
# print(xtest)
# print(ytest)

plt.imshow(mypet,cmap='gray')
plt.show()
cm = confusion_matrix(ytest,prediction)
print("confusion matrix")
print(cm)
spec=cm[1][1]/(cm[1][1]+cm[1][0])
print("specificity : ",spec)
prec=cm[0][0]/(cm[0][0]+cm[1][0])
print("precision : ",prec)
rec=cm[0][0]/(cm[0][0]+cm[0][1])
print("Recall : ",rec)
f1_score=(2*prec*rec)/(prec+rec)
print("F1 Score:",f1_score)
#use model to predict probability that given y value is 1
y_pred_proba = model.predict_proba(xtest)[:,1]

#calculate AUC of model
auc = metrics.roc_auc_score(ytest, y_pred_proba)
print(auc)
#print(cm[1][1])
#print(data)
