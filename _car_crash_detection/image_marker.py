import os
import pathlib

import numpy as np
import pandas as pd
import cv2
from PIL import Image

#draw and plot image

def draw_bounding_boxes(img, detection_boxes, detection_classes, detection_scores, num_detections, class_info):
    height, width, _ = img.shape
 
    for i in range(num_detections):

        box = detection_boxes[i]
        score = detection_scores[i]
        if(score < 0.5):
            continue
        cls = int(detection_classes[i])
        

        # draw bounding box
        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height)),
                            (int(box[3] * width), int(box[2] * height)), class_info[int(cls)][1], 8)
 
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

def draw_points(img, pose_scores, keypoint_scores):
    num_detections = 0
    for i in range(10):
        if pose_scores[i] != 0:
            num_detections += 1

    for i in range(num_detections):
        kc = pd.read_csv('C:/share/pose_estimation_outputs/' + image_name + '_keypoint_coords' + str(i) + '.csv')
        keypoint_coords = kc.to_numpy()

        for j in range(17):
            if(keypoint_scores[i][j] < 0.25):
                continue
            img = cv2.circle(img, 
                            (int(keypoint_coords[j][1]), int(keypoint_coords[j][0])), 
                            5, (0,0,0), -1)
    
    return img

#counting file number
count = 0
for path in pathlib.Path("C:/share/images").iterdir():
    if path.is_file():
        count += 1

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:/share/images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

for k in range(count):

    #read image
    
    feed_image = np.array(Image.open(TEST_IMAGE_PATHS[k]))
    image_name = 'image' + str(k)

    #load object detection csv data
    db = pd.read_csv('C:/share/object_detection_outputs/' + image_name + '_detection_boxes.csv')
    dc = pd.read_csv('C:/share/object_detection_outputs/' + image_name + '_detection_classes.csv')
    ds = pd.read_csv('C:/share/object_detection_outputs/' + image_name + '_detection_scores.csv')
    f = open('C:/share/object_detection_outputs/' + image_name + '_num_detections.csv', 'r')
    num_detections = int(f.read())
    f.close()
    #drop 0 column
    detection_boxes = db.iloc[:, 1:].to_numpy()
    detection_classes = dc.iloc[:, 1:].to_numpy()
    detection_scores = ds.iloc[:, 1:].to_numpy()

    #load pose estimation csv data
    #ps = pd.read_csv('C:/share/pose_estimation_outputs/' + image_name + '_pose_scores.csv')
    #ks = pd.read_csv('C:/share/pose_estimation_outputs/' + image_name + '_keypoint_scores.csv')

    #pose_scores = ps.to_numpy()
    #keypoint_scores = ks.to_numpy()


    class_info = {}
    f = open('class_info.txt', 'r')
    for line in f:
        info = line.split(', ')
    
        class_index = int(info[0])
        class_name = info[1]
        color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))    
        
        class_info[class_index] = [class_name, color]
    f.close()

    image = draw_bounding_boxes(feed_image, detection_boxes, detection_classes, detection_scores, num_detections, class_info)
    #image = draw_points(image, pose_scores, keypoint_scores)
    cv2.imwrite('C:/share/output_images/' + image_name + '.jpg', image)

print('drawing complete')