import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
dir="C:\\Users\\Admin\\Documents\\Classification_ML_2\\DATA\\DATASET\\DATASET\\TESTCAM"
categories=['O','R']
dataT=[]
for category in categories:
    path=os.path.join(dir,category)
    label=categories.index(category)
    for img in os.listdir(path):
        imgpath=os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        try:
            img = cv2.GaussianBlur(pet_img, (21, 21), 0)
            pet_img=cv2.resize(pet_img,(50,50))
            image=np.array(pet_img).flatten()
            dataT.append([image,label])
        except Exception as e:
            pass
pick_in=open('datas.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()
# #print(data)
random.seed( 20 )
print(len(data))

random.shuffle(data)
features=[]
labels=[]
for feature,label in data:
    features.append(feature)
    labels.append(label)
featuresT=[]
labelsT=[]
for feature,label in dataT:
    featuresT.append(feature)
    labelsT.append(label)
xtrainT,xtestT,ytrainT,ytestT=train_test_split(featuresT,labelsT,test_size=1)

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
prediction=model.predict(xtestT)
accuracy=model.score(xtestT,ytestT)
categories=['O','R']
print('Accuracy: ',accuracy)
print('prediction: ',prediction[0])
if(prediction[0]==1):
    print("prediction: Recycalble Waste")
else:
    print("prediction: Organic Waste")
mypet=xtestT[0].reshape(50,50)
# print(xtest)
# print(ytest)

plt.imshow(mypet,cmap='gray')
plt.show()
# cm = confusion_matrix(ytestT,prediction)
# print("confusion matrix")
#print(cm)
# spec=cm[0][0]/(cm[0][0]+cm[0][1])
# print("specificity : ",spec)
# sen=cm[0][0]/(cm[1][0]+cm[1][1])
# print("Sensitivity : ",sen)
# prec=cm[1][1]/(cm[1][1]+cm[0][1])
# print("precision : ",prec)
# rec=cm[1][1]/(cm[1][1]+cm[0][1])
# print("Recall : ",rec)
#print(cm[1][1])
#print(data)
