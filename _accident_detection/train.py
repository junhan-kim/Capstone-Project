# <issue>
# acc_det 가상환경 사용
# 루트 경로는 share.  (_accident_detection이 아님)
# 1. horizontal flip 활성화
# 2. batch size 에러 : 2의 배수로 변경하면 좋은데, 일단은 에러해결이 안되서 None으로 적용 중 (학습이 좀 느릴뿐이지 성능에는 큰 문제없을듯
#   => 문제는 batch size가 안정해져있다보니 한번에 gpu 메모리에 올라가는 데이터가 너무 많아서 가끔 ran out of memory 가 뜸..
# 3. h5가 아닌 pb로 save해야 하며, evaluate로 test셋을 통해 테스트할 수 있게끔
# 4. 기존 모델 삭제하고 실행

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import datasets
import IPython.display as display
from PIL import Image

from glob import glob

import pathlib

train_dir = pathlib.Path('./_accident_detection/images/train')
validation_dir = pathlib.Path('./_accident_detection/images/validation')
train_image_count = len(list(train_dir.glob('*/*.jpg')))
validation_image_count = len(list(validation_dir.glob('*/*.jpg')))
print(train_image_count)
print(validation_image_count)

CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES

accidents = list(train_dir.glob('accident/*'))

for image_path in accidents[:3]:
    display.display(Image.open(str(image_path)))


# horizontal flip
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True) # Generator for our validation data

BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160
STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)
VALIDATION_STEPS = np.ceil(validation_image_count/BATCH_SIZE)
print(STEPS_PER_EPOCH)
print(VALIDATION_STEPS)

train_data_gen = image_generator.flow_from_directory(directory='C:/Users/junhan_window/share/_accident_detection/images/train',   # str(train_dir)
                                                     batch_size=BATCH_SIZE,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(directory='C:/Users/junhan_window/share/_accident_detection/images/validation',   # str(validation_dir)
                                                              batch_size=BATCH_SIZE,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
                        
        
#모델
def create_model():
    IMG_SHAPE = (160,160,3)

    base_model = tf.keras.applications.DenseNet201(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=IMG_SHAPE,
        pooling='avg', classes=1000
    )

    base_model.trainable = True
    #base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    #tf.keras.layers.GlobalAveragePooling2D,
    tf.keras.layers.Dense(1920, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
    ])

    base_learning_rate = 0.0001
    model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #loss = 'categorical_crossentropy',
    metrics=['accuracy']
    )
    return model

model = create_model()

################### tpu test
'''
with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
  model = create_model()
'''
model.summary()

len(model.trainable_variables)

# batch_size : batch_size = BATCH_SIZE
# 현재는 batch size가 None인데 적절한 2의 배수로 바꿔줘야 함 (너무 작으면 결과값에 안좋은 영향, 너무 크면 학습속도 느림)
# 이 과정에서 데이터셋의 셔플링(shuffling)을 수행함
# epoch를 10으로 했는데도 과적합이 너무 높은거 같다는 느낌을 받음 (accuracy가 0.9973)
history = model.fit(train_data_gen, batch_size=None, validation_data=val_data_gen, epochs=1, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

'''
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
'''

# We can save our model with:
#model.save('./dense201_10epochs_model.hdf5')
model.save('.', save_format='tf')   # save_format='tf' 로 지정해야 test에 쓸 수 있는 pb파일이 나옴
