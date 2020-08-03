import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2
import streamlit as st
from sklearn.svm import SVC

from PIL import Image


def main():
    st.title("Gender Detector")
    # st.write(tensorflow.__version__)

    @st.cache(persist=True)
    def load_data():

        # pickle files
        mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
        model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
        model_pca = pickle.load(open('./model/pca_120.pickle', 'rb'))

        return mean, model_svm, model_pca

    mean, model_svm, model_pca = load_data()

    haar = cv2.CascadeClassifier(
        './model/haarcascade_frontalface_default.xml')

    gender_pre = ['Male', 'Female']
    font = cv2.FONT_HERSHEY_SIMPLEX

    def pipeline_model(img, color='rgb'):
        # step-2: convert into gray scale
        if color == 'bgr':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # step-3: crop the face (using haar cascase classifier)
        faces = haar.detectMultiScale(gray, 1.5, 3)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h),
                          (0, 255, 0), 2)  # drawing rectangle
            roi = gray[y:y+h, x:x+w]  # crop image
            # step - 4: normalization (0-1)
            roi = roi / 255.0
            # step-5: resize images (100,100)
            if roi.shape[1] > 100:
                roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)
            else:
                roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
            # step-6: Flattening (1x10000)
            roi_reshape = roi_resize.reshape(1, 10000)  # 1,-1
            # step-7: subptract with mean
            roi_mean = roi_reshape - mean
            # step -8: get eigen image
            eigen_image = model_pca.transform(roi_mean)
            # step -9: pass to ml model (svm)
            results = model_svm.predict_proba(eigen_image)[0]
            # step -10:
            predict = results.argmax()  # 0 or 1
            score = results[predict]
            # step -11:
            text = "%s : %0.2f" % (gender_pre[predict], score)
            cv2.putText(img, text, (x, y), font, 1, (0, 255, 0), 2)
        return img

    st.markdown("## Select an Image")
    uploaded_file = st.file_uploader(
        label="", type=['png', 'jpg', 'mp4', '#gp', 'webm', 'wmv'])
    if uploaded_file is not None:
        file_type = -1
        try:
            Image.open(uploaded_file)
            file_type = 1

        except:
            file_type = -1

        if(file_type == 1):
            image = Image.open(uploaded_file)
            img = np.array(image)
            img = pipeline_model(img)
            st.image(img, use_column_width=True)
        if(file_type == -1):
            st.markdown("### Unsupported file type")


if __name__ == '__main__':
    main()
