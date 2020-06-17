# Enter Interpreter path로 등록,   path확인하여 tensor2 사용
# 현재 가상환경 : base
# 절대 base랑 conda에 동일 패키지 같이 깔지말기. 충돌나면 pip conda에 있는 동일 패키지 다 삭제하고 pip에만 다시 설치

# pip install keras==2.2.4 하여 KeyError : 0 해결

# h5py 에러는 pip install h5py --upgrade --no-dependencies --force  로 해결

# damage_model = load_model(root + '_car_crash_detection/dense201_20epochs_6202_2071_model.h5')
# load_model로 h5파일을 load하되, 경로를 올바르게 작성.

import os
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

root = "C:/Users/junhan_window/share/"
image_path = root + '/_car_crash_detection/images'
output_dir = 'C:/Users/junhan_window/share/_web/static/input_images/'

def load_model_local(model_name):
  model_dir = root + '_car_crash_detection/object_detection_models/' + model_name + '/saved_model'

  model = tf.compat.v2.saved_model.load(str(model_dir), None)
  #model = tf.saved_model.load(model_dir)
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = root + '_car_crash_detection/object_detection/data/kitti_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path(image_path)
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

# select model (from api model zoo)
model_name = 'faster_rcnn_resnet101_kitti_2018_01_28'
selected_model = load_model_local(model_name)

#데미지 체크할 모델
damage_model = load_model(root + '_car_crash_detection/dense201_20epochs_6202_2071_model.h5') # dense201_20epochs_6202_2071_model.h5
#_model = root + '_car_crash_detection/damaged_model'
#damage_model = load_model(_model)  

#damage_model = tf.keras.models.load_model(_model)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  #print('run before')
  # Run inference
  output_dict = model(input_tensor)
  #print('run')

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

flag = False

count = 0
for path in pathlib.Path(image_path).iterdir():
  if path.is_file():
    count += 1

print(count)

check = 0
temp = 0
accident_flag = 0

start = time.time()
for i in range(count):
  #feed test image
  feed_image = np.array(Image.open(TEST_IMAGE_PATHS[i]))
  #print(feed_image)
  output = run_inference_for_single_image(selected_model, feed_image)
  #print(output)
  print('computed well')
  
  curcoord = list()
  prevcoord = list()
  diff = list()
  last_idx = list()

  if i == 0:
    prevoutput = output
  else:
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

              
  
  if temp == 1:
    temp = 0
    check += 1

  if i%10 == 0:
    if check > 9:
      print('stopped')
      last_img = Image.open(TEST_IMAGE_PATHS[i])
      width, height= last_img.size
      print (width, height)
      print(last_idx)
      
      for idx in last_idx:
            crd_x = int(output['detection_boxes'][idx][1] * width)
            crd_y = int(output['detection_boxes'][idx][0] * height)
            crd_h = int(output['detection_boxes'][idx][2] * height)-int(output['detection_boxes'][idx][0] * height)
            crd_w = int(output['detection_boxes'][idx][3]*width)-int(output['detection_boxes'][idx][1] * width)
            area = (crd_x,crd_y,crd_x+crd_w,crd_y+crd_h)
            size = crd_h*crd_w
            if size > 8000:       #너무 작은것들은 멀어서 좌표가 안바뀌는 것들 게다가 작은건 분별도 잘 못함 ㅠ 아마 픽셀을 강제로 늘여서 그런거같음
              print(area)
              cropped_img = last_img.crop(area)
              cropped_img.save(root + '/_car_crash_detection/cropped_images/' + 'croppedimage' + str(i) + '_' +str(idx) +  '.jpg')
                  
              
              test_img = cv2.imread(root + '/_car_crash_detection/cropped_images/' + 'croppedimage' + str(i) + '_' +str(idx) +  '.jpg')
              test_img = cv2.resize(test_img,(160,160), interpolation =  cv2.INTER_CUBIC)
              cv2.imwrite(root + '/_car_crash_detection/cropped_images/' + 'testimage' + str(i) + '_' +str(idx) +  '.jpg', test_img)
              test_img = np.array(test_img)
              test_img = np.expand_dims(test_img,axis=0)
              test_img = test_img/255
              #print(test_img.shape)
              
              accident = damage_model.predict_classes(test_img)

              
              if accident == 0 and flag == False:
                flag = True
                print("사고가 의심되는 상황입니다. 확인하여주십시오")
               
                cv2.imwrite(output_dir + 'image' + str(i) + '_.jpg', image)

                #여기에 동영상 만드는 코드
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                filename = output_dir + 'accident_video' + '.avi'
                out = cv2.VideoWriter(filename, fourcc, 5, (1280,720))
                start_i = i-50  # 최근 50장에 대해 저장
                if start_i < 0:
                  start_i = 0
                for cnt in range(start_i,i):
                  #이부분 경로는 연구실 컴퓨터에 맞게 수정해주세요
                  out.write(cv2.imread((output_dir + 'image'+str(cnt)+'.jpg'), cv2.IMREAD_COLOR))

                  if cv2.waitKey(1) == 27:
                    break
               
                out.release()

                #outputs zipping
                outputs_zip = zipfile.ZipFile(output_dir + 'object_detection_outputs.zip', 'w')
                
                for folder, subfolders, files in os.walk(root + '_car_crash_detection/object_detection_outputs'):
                
                    for file in files:
                        if file.endswith('.csv'):
                            outputs_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), root + '_car_crash_detection/object_detection_outputs'), compress_type = zipfile.ZIP_DEFLATED)
                
                outputs_zip.close()


                #동영상 만드는 코드 여기까지

                #cropped_img.show()


                #cropped_img.show()
              
              
              
              #cropped_img.show()
              #cv2.imwrite('C:/share/cropped_images/' + 'croppedimage' + str(i) + '.jpg', cropped_img)
      
        
      
    check = 0        

  prevoutput = output 
  print(i, check)

  #save output to csv
  output_name = 'image' + str(i)

  df = pd.DataFrame.from_dict(output['detection_classes'])
  df.to_csv(root + '/_car_crash_detection/object_detection_outputs/' + output_name + '_detection_classes.csv')
  df = pd.DataFrame.from_dict(output['detection_boxes'])
  df.to_csv(root + '/_car_crash_detection/object_detection_outputs/' + output_name + '_detection_boxes.csv')
  df = pd.DataFrame.from_dict(output['detection_scores'])
  df.to_csv(root + '/_car_crash_detection/object_detection_outputs/' + output_name + '_detection_scores.csv')
  df = output['num_detections']
  f = open(root + '/_car_crash_detection/object_detection_outputs/' + output_name + '_num_detections.csv', 'w')
  f.write(str(df))
  f.close()
  #print('output saved')

  #load label
  class_info = {}
  f = open(root + '_car_crash_detection/class_info.txt', 'r')
  for line in f:
      info = line.split(', ')
  
      class_index = int(info[0])
      class_name = info[1]
      color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))    
      
      class_info[class_index] = [class_name, color]
  f.close()

  #plot image
  image = draw_bounding_boxes(feed_image, output, class_info)
  #cv2.imwrite(root + '/_car_crash_detection/output_images/' + output_name + '.jpg', image)
  cv2.imwrite(output_dir + output_name + '.jpg', image)

print('Average FPS:', count / (time.time() - start))

