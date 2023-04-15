import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf
import os
import os
import glob
import time
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
st.set_page_config(layout="wide", page_title="Retinal disease diagnoser")
st.write("# major project by iffi zahid")
st.write("## Takes an OCT image as input and passes it through a custom pipeline containing Deep learning based image denoising, Segmentation and disease classification")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar :grin:"
)

st.sidebar.write("## Upload and download :gear:")



random_num = random.randint(10000,99999)

files_del = glob.glob('/home/crossfire/Programming projects/OCT/saves/*.png')
for f_del in files_del:
    os.remove(f_del)

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
    return pred


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def save_img(img):
    cv.imwrite("filename.png", img)

def thresh(orig, pred):
    img = cv.imread(orig, 0)       
    img = cv.resize(img, (180, 180))
    dst = cv.medianBlur(pred,1)
    blurred = cv.GaussianBlur(dst, (17,17), 0)
    _,th2 = cv.threshold(blurred,0.215,1,cv.ADAPTIVE_THRESH_MEAN_C)
    th2[th2!=0] = 255
    fin_img= np.multiply(th2,img)
    return fin_img

def load_prep(filename,img_shape=224):

    img=tf.io.read_file(filename)
    img=tf.image.decode_image(img,channels=3)
    img=tf.image.resize(img,size=[img_shape,img_shape])

    img=img/255.
    return img

def classify(orig):
    #orig = "/home/crossfire/Programming projects/OCT/im/"+upload.name
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    model_class = tf.keras.models.load_model("/home/crossfire/Programming projects/OCT/classify model.h5")
    classify=load_prep(orig)
    classify=tf.expand_dims(classify,axis=0)
    pred=model_class.predict(classify)
    result = pred.argmax()
    return class_names[result], pred[0]


# def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Original Image :camera:")
#     col1.image(image)

#     fixed = remove(image)
#     col2.write("Fixed Image :wrench:")
#     col2.image(fixed)
#     st.sidebar.markdown("\n")
#     st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

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
    seg_path = '/home/crossfire/Programming projects/OCT/saves/'+str(random_num)+'.png'
    plt.imsave(seg_path,predicted_img ,cmap='jet')
    return predicted_img, seg_path

def fix_image(upload):
    image = Image.open(upload)
    image= image.resize((500,500))
    col1.write("Original Image :camera:")
    col1.image(image)
    im_path = "/home/crossfire/Programming projects/OCT/im/"+upload.name
    fixed = denoise(im_path)
    fixed_new = cv.resize(fixed, dsize=(500, 500), interpolation=cv.INTER_AREA)
    col2.write("Denoised Image :wrench:")
    col2.image(fixed_new)
    thresh_img = thresh(im_path, fixed)
    thresh_new = cv.resize(thresh_img, dsize=(500, 500), interpolation=cv.INTER_AREA)
    col3.write("Thresholded image Image :camera:")
    col3.image(thresh_new, clamp=True, channels='gray' )
    _, seg_path = segment(im_path, fixed)
    #segmented = np.array(segmented, dtype=np.uint8)
    #segmented_new = cv.resize(segmented, dsize=(500, 500), interpolation=cv.INTER_AREA)
    seg_image = im.open(seg_path)
    col4.write("Segmented Image :wrench:")
    seg_image= seg_image.resize((500,500))
    col4.image(seg_image)
    pred_class, conf = classify(im_path)

    with st.sidebar:
        with st.spinner("Loading..."):
            time.sleep(1)
        st.success("Diagnosis: "+str(pred_class))
        st.success("Confidence: "+str("{:.2f}".format(max(conf)*100) )+"%")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2) 

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./zebra.jpg")


# st.sidebar.write(str(classify(im_path)))