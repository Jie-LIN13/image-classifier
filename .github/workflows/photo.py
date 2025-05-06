from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, layers
from PIL import Image
import numpy as np

## 資料目錄來源 以資料夾當作分類 , e.g. Datasets/cats, Datasets/dogs, ...
src_dir = r'Datasets/'
## 單次預測圖片
predict_img = r'Datasets/test.png'

## 影像讀取處理
datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
train_generator = datagen.flow_from_directory(src_dir, target_size=(224, 224), batch_size=20, subset='training')
valid_generator = datagen.flow_from_directory(src_dir, target_size=(224, 224), batch_size=20, subset='validation')

## 模型建立
mobilenetV2 = MobileNetV2(include_top=False, pooling='avg')
for mlayer in mobilenetV2.layers:
    mlayer.trainable = False
mobilenetV2output = mobilenetV2.layers[-1].output
fc = layers.Dense(units=train_generator.num_classes, activation='softmax', name='custom_fc') (mobilenetV2output)
classification_model = Model(
            inputs=mobilenetV2.inputs,
            outputs=fc)
classification_model.compile(loss='categorical_crossentropy', optimizer='Adam')

## 模型訓練
history = classification_model.fit(train_generator, epochs=10, validation_data=valid_generator)

## 單次預測
true_labels_dict = {}
for key in train_generator.class_indices:
    true_labels_dict[train_generator.class_indices[key]] = key

def pred(img_path):
    img = preprocess_input(np.array(Image.open(img_path).convert('RGB')))
    img = np.array([img])
    result_prob = classification_model.predict(img).tolist()[0]
    max_index = result_prob.index(max(result_prob))
    print(true_labels_dict[max_index], result_prob[max_index])

pred(predict_img)