# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

import boto3
from boto.s3.connection import S3Connection
from multiprocessing import cpu_count
conn = S3Connection()
s3bucket = conn.get_bucket('rekognitionbotoartifact')


def getdetector(model_folder_param='model'):
    detector = MtcnnDetector(model_folder=model_folder_param, ctx=mx.cpu(0), num_worker = cpu_count() , accurate_landmark = False)
    return detector

def countfaces(imagelocation, detector):
    image=imagelocation.replace('+',' ')
    #retrieve and print a selected image
    #imagesuffix=image[image.rfind('/')+1:]
    #print(imagesuffix)
    #folder='/home/ec2-user/demofolder/'
    chipcountpic='/home/ubuntu/MTCNN-face-filter/chip_count_pic.png'

    key = s3bucket.get_key(image)
    key.get_contents_to_filename(chipcountpic)
    #filename=folder+imagesuffix

    img = cv2.imread(chipcountpic)
    #print 'step1'
    # run detector
    results = detector.detect_face(img)
    #print 'step2'
    if results is not None:

        #total_boxes = results[0]
        return 'num_faces: '+str(len(results[0]))
        #print len(total_boxes)

    else:
        return 'num_faces: 0'
