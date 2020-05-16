import argparse
import cv2
import time
import glob
import random
import math
import numpy as np
import dlib
import itertools
from joblib import dump, load
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier


# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="svc/static")
a = ap.parse_args()
mode = a.mode 

# emotions = ["angry", "disgusted","fearful","happy","neutral","sad","surprised"] #fre
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file


# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, multi_class ='ovr', random_state = 0)
# clf = MLPClassifier(hidden_layer_sizes=(272,250,200,150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=0)

data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotion)
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotion)
    return training_data, training_labels, prediction_data, prediction_labels


if mode == "svc":
    clf = SVC(kernel='linear', probability=True, tol=1e-3, random_state = 0)
    print("Making sets...") #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("Training SVC classifier... ") #train SVM
    clf.fit(npar_train, training_labels)
    dump(clf, 'svc.joblib') 
    print("Getting accuracies... ") 
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("Accuracy: ", pred_lin)
    disp = plot_confusion_matrix(clf, prediction_data, prediction_labels, display_labels = emotions, cmap =plt.cm.Blues, normalize = 'true')
    disp.ax_.set_title("Confusion matrix - SVC")
    plt.show()
    count = [136, 55, 178, 76, 205, 83, 250]
    bars = ("anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, count)
    plt.xticks(y_pos, bars)
    plt.show()


elif mode == "static":
    clf = load('svc.joblib')
    cv2.ocl.setUseOpenCL(False)
    image = cv2.imread("static/sample.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/gray.jpg",gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    cv2.imwrite("static/clahe.jpg",clahe_image)
    detections = detector(clahe_image, 1)
    for k,d in enumerate(detections):
        temp_clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(temp_clahe_image, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 1)
        cv2.imwrite("static/clahe_face_detected.jpg", temp_clahe_image)
        shape = predictor(clahe_image, d)
        xlist = []
        ylist = []
        for i in range(1,68): 
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(temp_clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
        cv2.imwrite("static/clahe_facial_landmark.jpg",temp_clahe_image)   
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        start_point = (int(xmean), int(ymean))
        for i in range(1,68): 
            end_point = (shape.part(i).x, shape.part(i).y)
            cv2.line(temp_clahe_image, start_point, end_point, (0,255,0), 1)
        cv2.circle(temp_clahe_image, (int(xmean), int(ymean)), 1, (255,0,0), thickness=7)
        cv2.imwrite("static/clahe_facial_landmark_lines.jpg",temp_clahe_image)


