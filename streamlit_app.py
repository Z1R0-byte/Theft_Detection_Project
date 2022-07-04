import streamlit as st
import cv2
import tempfile
from ModelSTAE import loadModel
import numpy as np
import imutils

def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

model = loadModel()
model.load_weights('weights')


f = st.file_uploader("""#Upload video here""")

if f is not None: 
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())


    cap = cv2.VideoCapture(tfile.name)

    all_frames = []
    stframe = st.empty()

    while cap.isOpened():
        imagedump=[]
        ret,frame=cap.read()
        no_more_frames = False
        for i in range(10):
            ret,frame=cap.read()
            if(ret == True):
                image = imutils.resize(frame,width=700,height=600)
                frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean())/gray.std()
                gray=np.clip(gray,0,1)
                imagedump.append(gray)
            else: 
                no_more_frames = True
                break 
            
        imagedump=np.array(imagedump)
        imagedump.resize(227,227,10)
        imagedump=np.expand_dims(imagedump,axis=0)
        imagedump=np.expand_dims(imagedump,axis=4)
        output=model.predict(imagedump)
        loss=mean_squared_loss(imagedump,output)
        print(loss) 
        if no_more_frames: 
            break
        if frame.any() == None:
            print("none")
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        if loss>0.000418:
            print('Abnormal Event Detected')
            cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),3)
        all_frames.append(image)

    for img in all_frames: 
        stframe.image(img, channels='BGR')    

    cap.release()
    cv2.destroyAllWindows()