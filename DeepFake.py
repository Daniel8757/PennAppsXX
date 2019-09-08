
import os
import youtube_dl
from flask import Flask, redirect, render_template, send_from_directory, request
import os
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",  methods=['GET', 'POST'])
def upload():
    text = request.form['text']
    print(text)
   
    ex = "chmod 755 youtube-dl; ./youtube-dl "+text+" -o upload"
    os.system(ex)
    print("here")
    extractFrames("/Users/sofielysenko/Downloads/PennAppsXX-master-4/PennApps/upload.mkv")
    result = normalize("frames", "normalized")
    print(result)
    
    
    return render_template('return.html', response=result)

def extractFrames(path):
    #Extract frame every second
    print(cv2.__version__)
    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        cv2.imwrite("/Users/sofielysenko/Downloads/PennAppsXX-master-4/PennApps/frames/%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        count += 1



data_full = []
for i in range(2,12): #was 2-12
    data_full.append(i)


#getting frames from json (what is in the folder at this point)

#########Normalizing frames
# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#    help="path to facial landmark predictor")s
#ap.add_argument("-i", "--image", required=True,
#    help="path to input image")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner

def normalize(source_folder, final_folder):
    for i in range(len(data_full)):
        print(i)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("/Users/sofielysenko/shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceWidth=256)
        
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread("/Users/sofielysenko/Desktop/" + source_folder + "/" + str(data_full[i]) + ".jpg")
        try:
            image = imutils.resize(image, width=800)
        except:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # show the original input image and detect faces in the grayscale
        # image
        cv2.imshow("Input", image)
        rects = detector(gray, 2)
        
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            try:
                faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
                faceAligned = fa.align(image, gray, rect)
            except:
                continue
        
            import uuid
            f = str(uuid.uuid4())
            cv2.imwrite("foo/" + f + ".png", faceAligned)
            
            # display the output images
            cv2.imwrite("/Users/sofielysenko/Desktop/" + final_folder + "/%d.jpg" % data_full[i], faceAligned)
            cv2.imshow("Original", faceOrig)
            cv2.imshow("Aligned", faceAligned)

    return extractConcavityTest("concavity_test_obama.csv", final_folder)
#cv2.waitKey(0)

#normalize("real_obama_frames", "normalized_obama_frames")



#data_full = [0]

def extractConcavityTest(csvfile, norm_frames):   #extracting from the video we have online (using existing obama as testing later)
    
    import numpy as np
    from skimage import io
    
    import csv
    
    Image, Real, m_bottom, m_top, r_eye_top, l_eye_top, r_eye_bottom, l_eye_bottom = 'Real','Emotion', 'm_bottom', 'm_top', 'r_eye_top', 'l_eye_top', 'r_eye_bottom', 'l_eye_bottom'
    csvRow = [Image, Real, m_bottom, m_top, r_eye_top, l_eye_top, r_eye_bottom, l_eye_bottom]
    predictor_path = "/Users/sofielysenko/anaconda-backup/Documents/shape_predictor_68_face_landmarks.dat"
    
    
    myFile = open(csvfile, 'w')
    with myFile:
        myFields = ['Image','Real', 'm_bottom', 'm_top', 'r_eye_top', 'l_eye_top', 'r_eye_bottom', 'l_eye_bottom']
        writer = csv.DictWriter(myFile, fieldnames=myFields)
        writer.writeheader()

    for i in range(len(data_full)):
        try:
            img = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        dets = detector(img)
        
        RIGHT_EYE_TOP_POINTS = list(range(42, 46))
        LEFT_EYE_TOP_POINTS = list(range(42, 46))
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k, d in enumerate(dets):
            shape = predictor(img, d)
        
        vec = np.empty([68, 2], dtype = int)
        for b in range(68):
            vec[b][0] = shape.part(b).x #if you get a Name error for this line (shape not defined): try to change the first three lines of code to a more narrow range. The first couple images may not contain an infant from the video, or the infant's face is occluded
            vec[b][1] = shape.part(b).y #this error occurs because the array is empty due to the infant's face not being detected. I


        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]

        vecL = vec[54:61]
        vecR = vec[54:61]



        print(vecL)
        print(vecR)

#PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x, y = vecL.T
        #        plt.scatter(x,y)
        
        coeff = np.polyfit(x,y,2)
        
        xx = np.arange(min(x),max(x),.1)
        yy = xx**2*coeff[0] + xx*coeff[1] + coeff[2]
        
        #plt.plot(x,y, label='data')
        #        plt.plot(xx,yy, label='fitted function')
        #        plt.legend()
        #        plt.show()
        
        c = np.poly1d(coeff)
        crit = c.deriv().r
        r_crit = crit[crit.imag==0].real
        test = c.deriv(2)(r_crit)
        
        print(data_full[i])
        print("Concavity:")
        print(test)
        
        filename = data_full[i]
        concavity = test
        
        
        #np.savetxt('Run1.csv', test, fmt='%.8f', delimiter=',', header="Concavity")
        
        ######################################################
        try:
            img1 = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        
        detector1 = dlib.get_frontal_face_detector()
        predictor1 = dlib.shape_predictor(predictor_path)
        
        dets1 = detector(img1)
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k1, d1 in enumerate(dets1):
            shape1 = predictor1(img1, d1)
        
        vec1 = np.empty([68, 2], dtype = int)
        for b1 in range(68):
            vec1[b1][0] = shape1.part(b1).x
            vec1[b1][1] = shape1.part(b1).y
    
    
        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]
        
        vecL1 = vec1[48:55]
        vecR1 = vec1[48:55]
        
        
        
        print(vecL1)
        print(vecR1)
        
        #PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x1, y1 = vecL1.T
        #        plt.scatter(x1,y1)
        
        coeff1 = np.polyfit(x1,y1,2)
        
        xx1 = np.arange(min(x1),max(x1),.1)
        yy1 = xx1**2*coeff1[0] + xx1*coeff1[1] + coeff1[2]
        
        #plt.plot(x,y, label='data')
        #        plt.plot(xx1,yy1, label='fitted function')
        #        plt.legend()
        #        plt.show()
        #
        c1 = np.poly1d(coeff1)
        crit1 = c1.deriv().r
        r_crit1 = crit1[crit1.imag==0].real
        test1 = c1.deriv(2)(r_crit1)
        
        print(data_full[i])
        print("Concavity:")
        print(test1)
        
        filename1 = data_full[i]
        concavity1 = test1
        
        ###########################################################
        try:
            img2 = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        detector2 = dlib.get_frontal_face_detector()
        predictor2 = dlib.shape_predictor(predictor_path)
        
        dets2 = detector(img2)
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k2, d2 in enumerate(dets2):
            shape2 = predictor2(img2, d2)
        
        vec2 = np.empty([68, 2], dtype = int)
        for b2 in range(68):
            vec2[b2][0] = shape2.part(b2).x
            vec2[b2][1] = shape2.part(b2).y
        
        
        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]
        
        vecL2 = vec2[42:46]
        vecR2 = vec2[42:46]
        
        
        
        print(vecL2)
        print(vecR2)
        
        #PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x2, y2 = vecL2.T
        #        plt.scatter(x2,y2)
        
        coeff2 = np.polyfit(x2,y2,2)
        
        xx2 = np.arange(min(x2),max(x2),.1)
        yy2 = xx2**2*coeff2[0] + xx2*coeff2[1] + coeff2[2]
        
        #plt.plot(x,y, label='data')
        #        plt.plot(xx2,yy2, label='fitted function')
        #        plt.legend()
        #        plt.show()
        
        c2 = np.poly1d(coeff2)
        crit2 = c2.deriv().r
        r_crit2 = crit2[crit2.imag==0].real
        test2 = c2.deriv(2)(r_crit2)
        
        print(data_full[i])
        print("Concavity:")
        print(test2)
        
        filename2 = data_full[i]
        concavity2 = test2
        
        ###########################################################
        try:
            img3 = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        detector3 = dlib.get_frontal_face_detector()
        predictor3 = dlib.shape_predictor(predictor_path)
        
        dets3 = detector(img3)
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k3, d3 in enumerate(dets3):
            shape3 = predictor3(img3, d3)
        
        vec3 = np.empty([68, 2], dtype = int)
        for b3 in range(68):
            vec3[b3][0] = shape3.part(b3).x
            vec3[b3][1] = shape3.part(b3).y
        
        
        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]
        
        vecL3 = vec3[36:40]
        vecR3 = vec3[36:40]
        
        
        
        print(vecL3)
        print(vecR3)
        
        #PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x3, y3 = vecL3.T
        #        plt.scatter(x3,y3)
        
        coeff3 = np.polyfit(x3,y3,2)
        
        xx3 = np.arange(min(x3),max(x3),.1)
        yy3 = xx3**2*coeff3[0] + xx3*coeff3[1] + coeff3[2]
        
        #plt.plot(x,y, label='data')
        #        plt.plot(xx3,yy3, label='fitted function')
        #        plt.legend()
        #        plt.show()
        
        c3 = np.poly1d(coeff3)
        crit3 = c3.deriv().r
        r_crit3 = crit3[crit3.imag==0].real
        test3 = c3.deriv(2)(r_crit3)
        
        print(data_full[i])
        print("Concavity:")
        print(test3)
        
        filename3 = data_full[i]
        concavity3 = test3
        
        
        ###########################################################
        try:
            img4 = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        
        detector4 = dlib.get_frontal_face_detector()
        predictor4 = dlib.shape_predictor(predictor_path)
        
        dets4 = detector(img4)
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k4, d4 in enumerate(dets4):
            shape4 = predictor4(img4, d4)
        
        vec4 = np.empty([68, 2], dtype = int)
        for b4 in range(68):
            vec4[b4][0] = shape4.part(b4).x
            vec4[b4][1] = shape4.part(b4).y
        
        
        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]
        
        vecL4 = vec4[45:48]
        vecR4 = vec4[45:48]
        
        
        
        print(vecL4)
        print(vecR4)
        
        #PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x4, y4 = vecL4.T
        #        plt.scatter(x4,y4)
        
        coeff4 = np.polyfit(x4,y4,2)
        
        xx4 = np.arange(min(x4),max(x4),.1)
        yy4 = xx4**2*coeff4[0] + xx4*coeff4[1] + coeff4[2]
        
        #        #plt.plot(x,y, label='data')
        #        plt.plot(xx4,yy4, label='fitted function')
        #        plt.legend()
        #        plt.show()
        
        c4 = np.poly1d(coeff4)
        crit4 = c4.deriv().r
        r_crit4 = crit4[crit4.imag==0].real
        test4 = c4.deriv(2)(r_crit4)
        
        print(data_full[i])
        print("Concavity:")
        print(test4)
        
        filename4 = data_full[i]
        concavity4 = test4
        
        
        ###########################################################
        try:
            img5 = io.imread("/Users/sofielysenko/Desktop/"+ norm_frames +"/"+ str(data_full[i])+".jpg")
        except:
            continue
        
        detector5 = dlib.get_frontal_face_detector()
        predictor5 = dlib.shape_predictor(predictor_path)
        
        dets5 = detector(img5)
        
        
        #output face landmark points inside retangle
        #shape is points datatype
        #http://dlib.net/python/#dlib.point
        for k5, d5 in enumerate(dets5):
            shape5 = predictor5(img5, d5)
        
        vec5 = np.empty([68, 2], dtype = int)
        for b5 in range(68):
            vec5[b5][0] = shape5.part(b5).x
            vec5[b5][1] = shape5.part(b5).y
        
        
        #m_bottom : vec[54:61]
        #m_top : [48:55]
        #r_eye_top: [42:46]
        #l_eye_top: [36:40]
        #r_eye_bottom: [42:46]
        #l_eye_bottom: [36:40]
        
        vecL5 = vec5[39:42]
        vecR5 = vec5[39:42]
        
        
        
        print(vecL5)
        print(vecR5)
        
        #PLOT X AND Y
        import numpy as np
        from matplotlib import pyplot as plt
        
        x5, y5 = vecL5.T
        #        plt.scatter(x5,y5)
        
        coeff5 = np.polyfit(x5,y5,2)
        
        xx5 = np.arange(min(x5),max(x5),.1)
        yy5 = xx5**2*coeff5[0] + xx5*coeff5[1] + coeff5[2]
        
        #plt.plot(x,y, label='data')
        #        plt.plot(xx5,yy5, label='fitted function')
        #        plt.legend()
        #        plt.show()
        
        c5 = np.poly1d(coeff5)
        crit5 = c5.deriv().r
        r_crit5 = crit5[crit5.imag==0].real
        test5 = c5.deriv(2)(r_crit5)
        
        print(data_full[i])
        print("Concavity:")
        print(test5)
        
        filename5 = data_full[i]
        concavity5 = test5
        
        ###########################################
        #Print all tests here######################
        ###########################################
        
        import csv
        Image, Real, m_bottom, m_top, r_eye_top, l_eye_top, r_eye_bottom, l_eye_bottom = 'Image','Real', 'm_bottom', 'm_top', 'r_eye_top', 'l_eye_top', 'r_eye_bottom', 'l_eye_bottom'
        csvRow = [Image, Real, m_bottom, m_top, r_eye_top, l_eye_top, r_eye_bottom, l_eye_bottom]
        
        
        
        with open(csvfile, "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([data_full[i], 'Fake', test[0], test1[0], test2[0], test3[0], test4[0], test5[0]])

    return PCA(csvfile[:-4], "PCA_test_obama")

#extractConcavityTest("obama_real_concavity1.csv", "normalized_obama")

def PCA(source, final_PCA):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("/Users/sofielysenko/PennApps 2019/virtEnv1/" + source + ".csv")
    
    from sklearn.preprocessing import StandardScaler
    
    features = ['m_bottom', 'm_top', 'r_eye_top', 'l_eye_top', 'r_eye_bottom', 'l_eye_bottom']
    
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['Real']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                               , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
        
    finalDf = pd.concat([principalDf, df[['Real']]], axis = 1)
                               
    finalDf.to_csv(final_PCA)
                               
    return classify(final_PCA)


#PCA("obama_all_concavity", "Obama1.csv")

def classify(testing):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    training = pd.read_csv("/Users/sofielysenko/Desktop/Obama_training.csv")
    
    testing = pd.read_csv("/Users/sofielysenko/PennApps 2019/virtEnv1/" + testing)
    
    X_train, X_test, y_train, y_test = training.drop('Real', axis=1), testing[['principal component 1','principal component 2', 'principal component 3']], training['Real'], testing['Real']
    
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    
    import statistics
    final = statistics.mode(y_pred)
    
    return(final)



if __name__ == '__main__': #true if you run the script directly
    app.run(debug=True)



