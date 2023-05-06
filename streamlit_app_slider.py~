import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf
import os
import os
import time
import glob
from keras.preprocessing import image
import tensorflow as tf 
import keras 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
from PIL import Image as im
from tensorflow.keras.utils import normalize
import random
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide", page_title="Retinal disease diagnoser")
#st.set_page_config("Webb Space Telescope vs Original OCT Telescope", "🔭")

st.write("## Takes an OCT image as input and passes it through a custom pipeline containing Deep learning based image denoising, Segmentation and disease classification")


st.sidebar.write("## Upload and download :gear:")


random_num = random.randint(10000,99999)

files_del = glob.glob('/home/crossfire/Programming projects/OCT/saves1/denoise/*.png')
for f_del in files_del:
    os.remove(f_del)

files_del = glob.glob('/home/crossfire/Programming projects/OCT/saves1/segment/*.png')
for f_del in files_del:
    os.remove(f_del)

files_del = glob.glob('/home/crossfire/Programming projects/OCT/saves1/thresh/*.png')
for f_del in files_del:
    os.remove(f_del)

files_del = glob.glob('/home/crossfire/Programming projects/OCT/saves1/orig/*.png')
for f_del in files_del:
    os.remove(f_del)

orig_path = '/home/crossfire/Programming projects/OCT/saves1/orig/'+str(random_num)+'.png'
seg_path = '/home/crossfire/Programming projects/OCT/saves1/segment/'+str(random_num)+'.png'
den_path = '/home/crossfire/Programming projects/OCT/saves1/denoise/'+str(random_num)+'.png'
thresh_path = '/home/crossfire/Programming projects/OCT/saves1/thresh/'+str(random_num)+'.png'

def denoise(im_path):
    model_den = tf.keras.models.load_model("/home/crossfire/Programming projects/OCT/denoise model.h5")
    val_image = []
    img = tf.keras.utils.load_img(im_path, target_size=(180,180), color_mode= 'grayscale')
    img = tf.keras.utils.img_to_array(img)
    img = img/255
    val_image.append(img)
    train_df = np.array(val_image)
    test_test = []
    SIZE_X = 180
    SIZE_Y = 180
    img = cv.imread(im_path, 0)       
    img = cv.resize(img, (SIZE_Y, SIZE_X))
    pred= model_den.predict(train_df)
    pred=np.reshape(pred, (180,180))
    den_path = '/home/crossfire/Programming projects/OCT/saves1/denoise/'+str(random_num)+'.png'
    plt.imsave(den_path,pred ,cmap='gray')
    return pred, den_path


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def save_img(img):
    cv.imwrite("filename.png", img)

def load_prep(filename,img_shape=224):

    img=tf.io.read_file(filename)
    img=tf.image.decode_image(img,channels=3)
    img=tf.image.resize(img,size=[img_shape,img_shape])

    img=img/255.
    return img

def thresh(orig, pred):
    img = cv.imread(orig, 0)       
    img = cv.resize(img, (180, 180))
    #img = cv.normalize(img, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    dst = cv.medianBlur(pred,1)
    blurred = cv.GaussianBlur(dst, (17,17), 0)
    _,th2 = cv.threshold(blurred,0.215,1,cv.ADAPTIVE_THRESH_MEAN_C)
    th2[th2!=0] = 255
    #fin_img= np.multiply(th2,img)
    th2=np.reshape(th2, (180,180))
    thresh_path = '/home/crossfire/Programming projects/OCT/saves1/thresh/'+str(random_num)+'.png'
    plt.imsave(thresh_path,th2 ,cmap='gray')
    return thresh_path 

def classify(upload):
    orig = "/home/crossfire/Programming projects/OCT/im/"+upload.name
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    model_class = tf.keras.models.load_model("/home/crossfire/Programming projects/OCT/classify model.h5")
    classify=load_prep(orig)
    classify=tf.expand_dims(classify,axis=0)
    pred=model_class.predict(classify)
    result = pred.argmax()
    return class_names[result], pred[0]

def segment(orig, pred):
    model_seg = tf.keras.models.load_model('/home/crossfire/Programming projects/OCT/segmentation model.hdf5')
    test_test = []
    SIZE_X = 640
    SIZE_Y = 640
    n_classes= 9 
    img = cv.imread(orig, 0)       
    img = cv.resize(img, (SIZE_Y, SIZE_X))
    test_test.append(img)
    test_test = np.array(test_test)
    test_test = np.expand_dims(test_test, axis=3)
    test_test = normalize(test_test, axis=1)
    test1 = test_test[0]
    test_img_norm=test1[:,:,0][:,:,None]
    test=np.expand_dims(test_img_norm, 0)
    prediction = (model_seg.predict(test))
    predicted_img = np.argmax(prediction, axis=3)[0,:,:]
    seg_path = '/home/crossfire/Programming projects/OCT/saves1/segment/'+str(random_num)+'.png'
    plt.imsave(seg_path,predicted_img ,cmap='jet')
    return seg_path

def fix_image(upload):
    image = Image.open(upload)
    image= image.resize((500,500))
    im_path = "/home/crossfire/Programming projects/OCT/im/"+upload.name
    den_arr,_= denoise(im_path)
    thresh(im_path, den_arr)
    segment(im_path, den_arr)

def orig_img(upload):
    fix_image(upload)
    im_path = "/home/crossfire/Programming projects/OCT/im/"+upload.name
    img = cv.imread(im_path, 0)       
    img = cv.resize(img, (180, 180))
    orig_path = '/home/crossfire/Programming projects/OCT/saves1/orig/'+str(random_num)+'.png'
    plt.imsave(orig_path,img ,cmap='gray')
    return orig_path


my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# if my_upload is not None:
#     fix_image(upload=my_upload)
# else:
#     fix_image("./zebra.jpg")

    


orig_img(upload=my_upload)
pred_class, conf = classify(upload=my_upload)

st.header("OCT image preprocessing steps")

st.write("")
"Optical coherence tomography (OCT) is a noninvasive imaging method that uses reflected light to create pictures of the Retina. It can be used for early detection of various eye diseases"
st.write("")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
    st.markdown("### Original OCT Image")
    image_comparison(
        img1=orig_path,
        img2=orig_path,
        label1="Original OCT",
        label2="Original OCT",
    )

with col2:
    st.markdown("### Denoised OCT Image")
    image_comparison(
        img1=orig_path,
        img2=den_path,
        label1="Original OCT",
        label2="Denoised OCT",
    )

with col3:
    st.markdown("### Thresholded OCT Image")
    image_comparison(
        img1=orig_path,
        img2=thresh_path,
        label1="Original OCT",
        label2="Thresholded OCT",
    )

with col4:
    st.markdown("### Segmented OCT Image")
    image_comparison(
        img1=orig_path,
        img2=seg_path,
        label1="Original OCT",
        label2="Segmented OCT",
    )

st.sidebar.image(orig_path, use_column_width=True)

with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(1)
    st.success("Diagnosis: "+str(pred_class))
    st.success("Confidence: "+str("{:.2f}".format(max(conf)*100) )+"%")

