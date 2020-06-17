# 가상환경 acc_det(x)
# base

import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
from skimage import transform
from scipy import stats
from collections import Counter
import os
import time
import zipfile

BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160

buf = []
BUF_SIZE = 100   # 현재 버퍼 100
THRESHOLD = 0.5  # 현재 임계값 0.85

root = "C:/Users/junhan_window/share/"
output_dir = 'C:/Users/junhan_window/share/_web/static/input_images/'

def convert(np_image):
   #np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (IMG_WIDTH, IMG_HEIGHT, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

# 기존 buf에서 pop_front 이후 현재 프레임에서 predict된 cur(class)를 buf에 push_back
# 현재 buf의 0(accident) 에 대한 빈도 확률 > THRESHOLD 이면 true를 반환.
def is_accident_detected(cur):  
    cur = cur.tolist()
    #stats.mode(buf)  
    if len(buf) < BUF_SIZE:  # buf가 10 이하
        buf.append(cur)
        return False
    buf.pop(0)  # pop_front
    buf.append(cur)  # push_back
    #print('now buf: ' + str(buf))
    countDict = Counter(buf)
    freqNum = countDict[0]  # buf내에서 0의 빈도수
    print('freqNum: ' + str(freqNum))
    if float(freqNum) / BUF_SIZE > THRESHOLD:
        return True
    else:
        return False


model = tf.keras.models.load_model('.')

# 1,3,4,6 에서 잘 잡음 (데이터셋 및 모델은 fire, epoch1)
cap = cv2.VideoCapture('./_accident_detection/fire6.mp4')

# frame이 너무 많아서 너무 오랫동안 predict가 지속됨. (frame을 건너뛸 필요가 있음)
# 1분 24초에 30 프레임 = 84 x 30 = 2520 
start = time.time()

cnt = -1
while(cap.isOpened()):
    ret, frame = cap.read()

    cnt += 1       
    #if cnt % 30 != 0:  # 프레임 건너뛰기 용 (지나치게 영상 길이가 길때 사용)
    #    continue

    img = convert(frame)
    print()
    print('now frame: ' + str(cnt))
    now_class = model.predict_classes(img)
    now_class = np.squeeze(now_class)  # 쓸모없는 차원 제거
    #print('now class: ' + str(now_class))    
    if is_accident_detected(now_class):
        print('accident detected !!!' + ' in frame' + str(cnt))
        #os.system("pause")
        cv2.imwrite(output_dir + 'image_f' + str(cnt) + '_.jpg', frame)

        #여기에 동영상 만드는 코드
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        filename = output_dir + 'accident_video_f' + '.avi'
        out = cv2.VideoWriter(filename, fourcc, 5, (640,360))
        start_i = cnt-50  # 최근 50장에 대해 저장
        if start_i < 0:
            start_i = 0
        for j in range(start_i,cnt):
            #이부분 경로는 연구실 컴퓨터에 맞게 수정해주세요
            out.write(cv2.imread((output_dir + 'image_f'+str(j)+'.jpg'), cv2.IMREAD_COLOR))

            if cv2.waitKey(1) == 27:
                break
        
        out.release()

        #outputs zipping
        outputs_zip = zipfile.ZipFile(output_dir + 'object_detection_outputs.zip', 'w')
        
        for folder, subfolders, files in os.walk(root + '_accident_detection/output_images'):
        
            for file in files:
                if file.endswith('.jpg'):
                    outputs_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), output_dir + 'object_detection_outputs'), compress_type = zipfile.ZIP_DEFLATED)
        
        outputs_zip.close()


        break

    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # opencv는 BGR 기반
    cv2.imwrite(output_dir + 'image_f'+str(cnt)+'.jpg', frame)
    #cv2.imshow('frame', rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Average FPS:', cnt / (time.time() - start))

cap.release()
cv2.destroyAllWindows()



'''
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = image_generator.flow_from_directory(directory='C:/Users/junhan_window/share/_accident_detection/images/test',
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

filenames = test_data_gen.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_data_gen, steps = nb_samples)
print(predict)
print(predict.shape)
'''
# model.evaluate(test_data_gen)

'''
from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (IMG_WIDTH, IMG_HEIGHT, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

for i in range(197, 511):
#for i in range(151, 197):
#for i in range(1000,2101):
    image = load('C:/Users/junhan_window/share/_accident_detection/images/test/accident/test용 ' + str(i) + '.jpg')
    #image = load('C:/Users/junhan_window/share/_accident_detection/images/test/noaccident/test용 ' + str(i) + '.jpg')
    #image = load('C:/Users/junhan_window/share/_accident_detection/images/train/noaccident/' + str(i) + '.jpg')
    print(model.predict_classes(image))
    #print(model.predict(image))
'''


# accident : 0, noaccident : 1

# 아직 학습이 덜 된듯
# evaluate 시 accuracy가 높게 나오는 이유 : accident던 noaccident던 둘 다 0으로 분류함.
# 근데 accident 데이터셋이 훨씬 많아서 확률이 높게 나오는 것..
# 두가지 대안 : 1. 학습을 더 많이 시켜서 구분 가능하게 해야함. 
#              2. 그냥 train set과 유사한 영상 또는 동일 영상을 test set으로 사용함

# <구현 프로세스>
# 1. 영상 데이터를 이미지 배열로 input
# 2. 이미지 배열의 각 이미지에 순차적으로 접근하여 model.predict_classes
# 3. 특정 구간에서 accident에 해당하는 binary 에 대한 frequency가 일정 threshold 이상의 빈도를 보일 경우
# 4. 그 임계값을 넘은 시점의 프레임 번호를 accident가 발생한 프레임 번호로 규정하고, 이 int 값을 렌더러 프로세스로 넘김