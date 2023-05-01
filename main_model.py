import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
model= tf.keras.models.load_model('./static/model/object_detection.h5')
def object_detect(path, filename):
    #read img
    image = load_img(path) #PIL Object
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path,target_size=(224,224))
    #data preprocess
    image_arr_224 = img_to_array(image1)/255.0
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    #makepredict
    coords = model.predict(test_arr)
    #denormalize the valuens
    denom = np.array([w,w,h,h]) 
    coords = coords * denom
    coords = coords.astype(np.int32)
    #draw
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,255),3)
    #convert bgr
    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords 