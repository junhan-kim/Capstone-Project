# base는 껏다키면 가끔 나옴. 안나오면 conda base 경로로 interpreter 추가
# 나머지 가상환경도 junhan_window 에 anaconda env 로 가서 추가


# tensor2(x)
# conda 꼬인듯 일단 pip으로 설치하고 base로 ㄱ

'''
4/16
경로관련 문제 다수 존재
(defalt path: C:/share 환경에서 개발, 제 컴퓨터에선 잘 돌아요)
gpu사용 관련 이슈 미해결
(internal: invoking ptxas not supported on windows relying on driver to perform ptx compilation. this message will be only logged once.)
api issue 사이트 참고
(https://github.com/tensorflow/models/issues/7640)


현재 기능
images folder의 하나의 이미지를 대상으로 연산
object_detection_output에 output csv파일 형태로 저장
이미지 plot

model: ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
label: mscoco_complete_label_map.pbtxt

4/19
1. missing path 추가 및 모델은 인터넷에서 별도로 다운로드
2. terminal에서 가상환경 활성화 -> vscode 껏다켜야 목록에 생김
   이후 shift-ctrl-p -> python select interpreter로 tensor2 선택
   tensor2에 tf2.0 설치
3. terminal이 아닌 별도 console에서 pip install 시도
4. window tensor 2.0 installation 참고 (protobuf 3.11.0 win64로 컴파일)
5. coco를 kitti 데이터셋(car and pedestrian) 기반 모델로 변경. class_info.txt에서 레이블 수정 
'''

# === 설치/실행 전 유의사항 ===
# 1. 좌측하단에서 tensor2 활성화.  
# 2. 단, terminal에선 가상환경 전환이 안되므로 cmd에서 activate tensor2 하여 
#    패키지 체크(conda list) 및 환경 설치(conda install ~). // cmd는 관리자권한 필수
# 3. _object_detection => tensor2
#    _pose_estimation => tensorjs
#      // 만일 tensor2 환경에 tensorflowjs를 다시 깔게 되면 tensorflow, IPython 다운그레이드되므로 재설치 요망
#      // 두 환경 같이 쓰면 충돌나므로 안됨 (tensorflow 2.x vs tensorflow 1.x)
#    _accident_detection => acc_det
#      // acc_det의 경우, ipynb 파일을 열고, 좌측하단이 아닌 우측상단에서 환경을 선택해줘야 함
# 4. 파이썬 가상환경 생성시에, conda create -n <이름> python=3.7   // 파이썬 버전 부여해야 vscode interpreter 목록에 뜸
# 5. 가상환경 변경시에는, 좌측 하단에서도 변경하고 + 터미널에서 conda activate <환경이름>

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
import time

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

root = 'C:/Users/junhan_window/share/'
root_dir = 'C:/Users/junhan_window/share/_object_detection/'

def load_model_local(model_name):
  model_dir = root_dir + 'object_detection_models/' + model_name + '/saved_model'
 
  #model = tf.saved_model.load(model_dir)
  model = tf.compat.v2.saved_model.load(str(model_dir), None)
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = root_dir + 'object_detection/data/mscoco_complete_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = pathlib.Path(root + 'images')
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
#TEST_IMAGE_PATHS

# select model (from api model zoo)
model_name = 'faster_rcnn_resnet101_kitti_2018_01_28'
selected_model = load_model_local(model_name)

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
f = open(root_dir + 'class_info.txt', 'r')
for line in f:
    info = line.split(', ')

    class_index = int(info[0])
    class_name = info[1]
    color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))    
    
    class_info[class_index] = [class_name, color]
f.close()

cap = cv2.VideoCapture(root + 'input_video/fire1.mp4')
cnt = -1
while(cap.isOpened()):
  if cnt==100:  # 임시로 설정
    break
  cnt += 1       
  print(cnt)

  ret, frame = cap.read()     
  output = run_inference_for_single_image(selected_model, np.array(frame))
  #print(output)

  #save output to csv
  output_name = 'image' + str(cnt)

  df = pd.DataFrame.from_dict(output['detection_classes'])
  df.to_csv(root_dir + 'object_detection_outputs/' + output_name + '_detection_classes.csv')
  df = pd.DataFrame.from_dict(output['detection_boxes'])
  df.to_csv(root_dir + 'object_detection_outputs/' + output_name + '_detection_boxes.csv')
  df = pd.DataFrame.from_dict(output['detection_scores'])
  df.to_csv(root_dir + 'object_detection_outputs/' + output_name + '_detection_scores.csv')
  df = output['num_detections']
  f = open(root_dir + 'object_detection_outputs/' + output_name + '_num_detections.csv', 'w')
  f.write(str(df))
  f.close()
  f = open(root_dir + 'object_detection_outputs/' + output_name + '_completed.txt', 'w')
  f.write("completed")
  f.close()
  #print('output saved')

  #plot image
  #image = draw_bounding_boxes(frame, output, class_info)
  #cv2.imwrite(root_dir + 'object_detection_outputs/' + output_name + '.jpg', image)

  #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #cv2.imshow('image', rgb)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()