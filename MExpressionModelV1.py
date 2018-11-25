import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#Code for function plots obtained from video: 
#https://www.youtube.com/watch?v=bfQBPNDy5EM

def plots(ims,figsize=(12,60),rows=1,interp=False,titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows+1
    for i in range(len(ims)):
        sp = f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else 'none')
#Preparation of Training, Validation, and Test Data
trPath = 'Data\\train'
tePath = 'Data\\test'
vaPath = 'Data\\valid'
cl = ['-','+','div','times','0','1','2','3','4','5','6','7','8','9']
##trBatches = ImageDataGenerator().flow_from_directory(trPath,
#                              target_size=(45,45),classes=cl,
#                              batch_size=1787)
##vaBatches = ImageDataGenerator().flow_from_directory(vaPath,
#                              target_size=(45,45),classes=cl,
#                              batch_size=56)
teBatches = ImageDataGenerator().flow_from_directory(tePath,
                              target_size=(45,45),class_mode=None,
                              batch_size=1)
##========= The Model =======##
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(45,45,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit_generator(trBatches,steps_per_epoch=88,validation_data=vaBatches,
                   # validation_steps=4,epochs=4,verbose=1)
teBatches.reset()
test_imgs = next(teBatches)
plots(test_imgs)
prediction = np.argmax(model.predict_generator(teBatches,steps=1,verbose=0))
print (cl[prediction])
