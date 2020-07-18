from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,MaxPooling2D,Reshape,LSTM
from keras.models import Model,Sequential,load_model
from sklearn.preprocessing import LabelBinarizer
from HandTracking import handtracking
import numpy as np
import warnings
import cv2
import os

warnings.filterwarnings("ignore", category=FutureWarning)

def vid2frames(vid_dir, frame_limit):

    """
    Isolate hands and from video and split video into individual frames
        
        PARAMETERS:
        vid_dir (str): Directory of videos

        frame_limit (int): Maximum number of frames per video
        
        RETURN:
        X_train: Training Frames

        Y_train: Labels of each frame
    
    """
    
    data = []
    
    labels = os.listdir(vid_dir)
        
    
    if ".DS_Store" in labels or "__MACOSX" in labels:
        labels.remove(".DS_Store")
    
    for label in labels:
        print("Processing ", label)
        
        videos = os.listdir(vid_dir+'/'+label)
        
        if ".DS_Store" in videos or "__MACOSX" in videos:
            videos.remove(".DS_Store")
        
        for vid in videos:

            i=0
            cap = cv2.VideoCapture(vid_dir+'/'+label+'/'+vid)
            while(cap.isOpened() and i<frame_limit):
                
                ret, frame = cap.read()
                
                if ret == False:
                    break
                    
                image = frame
                result = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                lower = np.array([41, 156, 49])
                upper = np.array([179, 255, 255])

                mask = cv2.inRange(image, lower, upper)
                result = cv2.bitwise_and(result, result, mask=mask)
                
                result = cv2.resize(result, (256,256))
                
                data.append([result, label])
                
                i += 1 
    
    X_train = []
    Y_train = []
    
    for result,label in data:
        X_train.append(result)
        Y_train.append(label)
        
    return(X_train, Y_train)

#X_train, Y_train = vid2frames('/content/VideoFiles', 100)

#encoder = LabelBinarizer()

#Y_encoded_train =  encoder.fit_transform(Y_train)
#X_train = np.array(X_train)
'''
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (256,256,3)))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Reshape((16, 16*256)))

model.add(LSTM(64, return_sequences=True, input_shape=(1,256)))

model.add(LSTM(32, return_sequences=True))

model.add(LSTM(32))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
'''
#model.fit(X_train, Y_encoded_train, epochs=5, batch_size=32)

'''
Epoch 1/5
29340/29340 [==============================] - 140s 5ms/step - loss: 0.3043 - accuracy: 0.8925
Epoch 2/5
29340/29340 [==============================] - 131s 4ms/step - loss: 0.1965 - accuracy: 0.9282
Epoch 3/5
29340/29340 [==============================] - 130s 4ms/step - loss: 0.2065 - accuracy: 0.9241
Epoch 4/5
29340/29340 [==============================] - 131s 4ms/step - loss: 0.1980 - accuracy: 0.9252
Epoch 5/5
29340/29340 [==============================] - 130s 4ms/step - loss: 0.1518 - accuracy: 0.9437
<keras.callbacks.callbacks.History at 0x7fd15e99b080>
'''

#model.save("modelv2.h5")

model = load_model('/Users/zafirkhalid/Desktop/BayHacks/modelv2.h5')

def predict_video():
  repeat = True

  while repeat:
    repeat = False
    handtracking()

    textscreen = np.zeros((512,512,3))
    cap = cv2.VideoCapture('/Users/zafirkhalid/Desktop/BayHacks/testvideo.mp4')

    i=0
    testdata = []
    predictions = []
    imgdata = []


    while(cap.isOpened() and i<=100):

        ret, frame = cap.read()

        if ret == None:
            break

        image = frame
        try:
            result = image.copy()
        except:
            break

        result = cv2.resize(result, (256,256))

        imgdata.append(result)

        if len(imgdata) == 16:
            testdata.append(imgdata)
            imgdata = []

        i += 1

    testdata = np.array(testdata)
    
    #testdata = testdata.squeeze()

    print(testdata[0].shape)
    for data in testdata:
        predictions.append((np.argmax(model.predict(data), axis=-1)))
    print(predictions)

    cv2.putText(textscreen,"Press Space to Restart or Q to Quit", (0,500), 0, 0.75, (255,255,255), 2, cv2.LINE_AA)
    
    for j in range(len(predictions)):
        result = np.bincount((predictions[j])).argmax()


        if result == 0:
            cv2.putText(textscreen,"Bright", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        elif result == 1:
            cv2.putText(textscreen,"Green", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        elif result == 2:
            cv2.putText(textscreen,"Light-Blue", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        elif result == 3:
            cv2.putText(textscreen,"Opaque", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        elif result == 4:
            cv2.putText(textscreen,"Red", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        elif result == 5:
            cv2.putText(textscreen,"Yellow", ((50+(j*100)),50), 0, 1, (255,255,255), 2, cv2.LINE_AA)


    while True: 
        cv2.imshow("Sample Text",textscreen)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key % 256 == 32:
            repeat = True
            break

predict_video()


#---DONE---#
#Add text display on window
#Add restart button


#---TO DO---#
#Add multiword functionality
#Word to sign conversion option