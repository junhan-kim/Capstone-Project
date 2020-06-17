import os
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
import pandas as pd

from tensorflow.keras.models import load_model

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model_local(model_name):
  model_dir = 'C:/share/object_detection_models/' + model_name + '/saved_model'
 
  model = tf.saved_model.load(model_dir)
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/share/object_detection/data/mscoco_complete_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:/share/images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

# select model (from api model zoo)
model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
selected_model = load_model_local(model_name)

#데미지 체크할 모델
damage_model = load_model('dense201_20epochs_6202_2071_model.h5')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=3)

  display(Image.fromarray(image_np))

count = 0
for path in pathlib.Path("C:/share/images").iterdir():
    if path.is_file():
        count += 1

check = 0
temp = 0
last_idx = 0
#check_flag = 0



for i in range(count):
  #feed test image
  feed_image = np.array(Image.open(TEST_IMAGE_PATHS[i]))
  output = run_inference_for_single_image(selected_model, feed_image)
  #print(output)
  #print('computed well')
  
  diff = list()
  curcoord = list()
  prevcoord = list()
  last_idx = list()
  
  if i == 0:
    prevoutput = output
  else:
    del last_idx[:]
    for previ in range(prevoutput['num_detections']):
      if prevoutput['detection_scores'][previ] > 0.5:
        prevcoord = prevoutput['detection_boxes'][previ]

        for curi in range(output['num_detections']):
          if output['detection_scores'][curi] > 0.5:
            curcoord = output['detection_boxes'][curi]
            diff = abs(curcoord - prevcoord)
            
            if diff[0] < 0.01 and diff[1] < 0.01:
              temp = 1
              last_idx.append(curi)
  
  
  '''
  if i==0:        #모두 담아야함
    prevoutput = output
    for previ in range(prevoutput['num_detections']): #모두 다 보고 신뢰도 0.5 이상이면
      if prevoutput['detection_scores'][previ] > 0.5:
        print("왜 안돼")  
        prevcoord.append(prevoutput['detection_boxes'][previ])  #curcoord
    
  else:
    print(prevcoord)
    for curi in range(output['num_detections']):
          if output['detection_scores'][curi] > 0.5:
            for coord in prevcoord:
              diff = abs(output['detection_boxes'][curi] - coord)
              if diff[0] < 0.01 and diff[1] < 0.01:
                if check_flag == 0:
                  check_flag = 1  #1개의 이미지에서 check가 1만 올라가기 위함
                  check += 1
                print("짜증나게")  
                curcoord.append(output['detection_boxes'][curi])
                #last_idx = curi #다 돌고 라스트 인덱스도 리스트로 저장
                last_idx.append(curi)

  print(prevcoord)
  print(curcoord)
  '''
           
              
  
  if temp == 1:
    temp = 0
    check += 1
  

  if i%10 == 0:
    if check > 9:
      print('stopped')
      
      
      last_img = Image.open(TEST_IMAGE_PATHS[i])
      width, height= last_img.size
      print (width, height)
      for idx in last_idx:
        crd_x = int(output['detection_boxes'][idx][1] * width)
        crd_y = int(output['detection_boxes'][idx][0] * height)
        crd_h = int(output['detection_boxes'][idx][2] * height)-int(output['detection_boxes'][idx][0] * height)
        crd_w = int(output['detection_boxes'][idx][3]*width)-int(output['detection_boxes'][idx][1] * width)
        
        '''
        #높이가 더 높은경우
        if(crd_h>crd_w):
          crd_bigger = crd_h
          crd_smaller = crd_w
          area = (crd_x-(crd_bigger/2),crd_y,crd_x+(crd_bigger/2),crd_y+crd_bigger)

        #가로가 더 긴경우
        else:
          crd_bigger =crd_w
          crd_smaller = crd_h
          area = (crd_x,crd_y-(crd_bigger/2),crd_x,crd_y+(crd_bigger/2))

        area = (crd_x,crd_y,crd_x+crd_bigger,crd_y+crd_bigger)
        
        if(crd_x+crd_bigger > width or crd_y + crd_bigger > height):
          area = (crd_x,crd_y,crd_x+crd_smaller,crd_y+crd_smaller)
        '''

        area = (crd_x,crd_y,crd_x+crd_w,crd_y+crd_h)
        
        #last_img.show()
        cropped_img = last_img.crop(area)
        cropped_img.save('C:/share/cropped_images/' + 'croppedimage' + str(i) + '.jpg')
        test_img = cv2.imread('C:/share/cropped_images/' + 'croppedimage' + str(i) + '.jpg')
        test_img = cv2.resize(test_img,(160,160), interpolation =  cv2.INTER_CUBIC)
        cv2.imwrite('C:/share/cropped_images/' + 'croppedimage' + str(i) +' ' +str(idx) +  '.jpg', test_img)
        test_img = np.array(test_img)
        test_img = np.expand_dims(test_img,axis=0)

        print(damage_model.predict_classes(test_img))
      

      
    check = 0        

  #check_flag  = 0
  #prevoutput = output 
  '''
  if i>1:
    prevcoord = curcoord[:] #리스트 복사해야
  print(prevcoord)
  del curcoord[:] #curcoord리스트 비우기
  del last_idx[:] #매 이미지마다 last_idx 비우기
  '''
  print(i, check)

  #save output to csv
  output_name = 'image' + str(i)

  df = pd.DataFrame.from_dict(output['detection_classes'])
  df.to_csv('C:/share/object_detection_outputs/' + output_name + '_detection_classes.csv')
  df = pd.DataFrame.from_dict(output['detection_boxes'])
  df.to_csv('C:/share/object_detection_outputs/' + output_name + '_detection_boxes.csv')
  df = pd.DataFrame.from_dict(output['detection_scores'])
  df.to_csv('C:/share/object_detection_outputs/' + output_name + '_detection_scores.csv')
  df = output['num_detections']
  f = open('C:/share/object_detection_outputs/' + output_name + '_num_detections.csv', 'w')
  f.write(str(df))
  f.close()
  #print('output saved')

  #draw and plot image
  def draw_bounding_boxes(img, output_dict, class_info):
      height, width, _ = img.shape
  
      obj_index = output_dict['detection_scores'] > 0.5
      
      scores = output_dict['detection_scores'][obj_index]
      boxes = output_dict['detection_boxes'][obj_index]
      classes = output_dict['detection_classes'][obj_index]
  
      for box, cls, score in zip(boxes, classes, scores):
          # draw bounding box
          img = cv2.rectangle(img,
                              (int(box[1] * width), int(box[0] * height)),
                              (int(box[3] * width), int(box[2] * height)), class_info[cls][1], 8)
  
          # put class name & percentage
          object_info = class_info[cls][0] + ': ' + str(int(score * 100)) + '%'
          text_size, _ = cv2.getTextSize(text = object_info,
                                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale = 0.9, thickness = 2)
          img = cv2.rectangle(img,
                              (int(box[1] * width), int(box[0] * height) - 25),
                              (int(box[1] * width) + text_size[0], int(box[0] * height)),
                              class_info[cls][1], -1)
          img = cv2.putText(img,
                            object_info,
                            (int(box[1] * width), int(box[0] * height)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
          
      return img

  #load label
  class_info = {}
  f = open('class_info.txt', 'r')
  for line in f:
      info = line.split(', ')
  
      class_index = int(info[0])
      class_name = info[1]
      color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))    
      
      class_info[class_index] = [class_name, color]
  f.close()

  #plot image
  image = draw_bounding_boxes(feed_image, output, class_info)
  cv2.imwrite('C:/share/output_images/' + output_name + '.jpg', image)
