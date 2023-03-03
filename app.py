import webbrowser

import cv2
import os
import requests
from flask import Flask, request, render_template, jsonify, request, json
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, storage
from urllib3.util import response
from flask_cors import CORS

config = {

    "apiKey": "AIzaSyC8ZhPCUc_rdB8ujfr9kvL1XV-d3bf08n4",

    "authDomain": "smart-healthcare-system-666e1.firebaseapp.com",

    "projectId": "smart-healthcare-system-666e1",

    "storageBucket": "smart-healthcare-system-666e1.appspot.com",

    "messagingSenderId": "895658645483",

    "appId": "1:895658645483:web:87e1593fe4edb3b9a8ebae"

}

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, config)

#### Defining Flask App
app = Flask(__name__)
CORS(app)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
# def totalreg():
#     return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
#don't put webbrrowser here
        if cv2.waitKey(1) == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

    webbrowser.open(f'http://localhost:3000/user/{identified_person}')



#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    # facedt =request.data.decode('utf-8')
    # newusername = facedt
    # facedt = request.get_json()
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        print (json['nic'])
    else:
        return 'Content-Type not supported!'
    # return facedt
    # stringdt = json.dumps(facedt)
    # return stringdt
    # data_str = json.dumps(facedt, indent=4)
    # print(data_str)
    # newusername = data_str
    # newusername = request.form['name']
    # newuserid = request.form['nic']
    newuserid = json['nic']
    # userimagefolder = 'static/faces/' + newusername
    userimagefolder = 'static/faces/' + str(newuserid)
    # userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newuserid + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
                bucket = storage.bucket()
                file = userimagefolder + "/" + name
                blob = bucket.blob(file)
                blob.upload_from_filename(file)
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()

    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    # names, rolls, times, l = extract_attendance()
    # return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)


#### Our main function which runs the Flask App


if __name__ == '__main__':
    app.run(debug=True)
