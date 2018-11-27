
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from PIL import Image
import cv2
import os
import numpy as np
from random import shuffle
from random import randint
from tqdm import tqdm
import matplotlib.pyplot as plt
trainData = 'Data\\train'
testData = 'test'
labels = ['-','+','div','times','0','1','2','3','4','5','6','7','8','9']
def oneHotLabel(label):
    ohl = [0]*len(labels)
    ohl[labels.index(label)] = 1
    return np.array(ohl)
def labeledTestData():
    testImg = []
    for x in tqdm(os.listdir(testData)):
        path = os.path.join(testData,x)
        for y in os.listdir(path):
            img = cv2.imread(os.path.join(path,y),cv2.IMREAD_GRAYSCALE)
            testImg.append([np.array(img),oneHotLabel(x)])
    return testImg
testImages = labeledTestData()
trData = np.load('trData.npy')
trLabels = np.load('trLabels.npy')
teData = np.array([i[0] for i in testImages]).reshape(-1,45,45,1)
teLabels = np.array([i[1] for i in testImages])

model = load_model('MExpressionModelV2_2(Heiarchy).hd5')
fig = plt.figure(figsize=(14,14))
for cnt,data in enumerate(testImages):
    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,45,45,1)
    modelOut = model.predict([data])
    strLabel = labels[(np.argmax(modelOut))]
    y.imshow(img,cmap='gray')
    plt.title(strLabel)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)