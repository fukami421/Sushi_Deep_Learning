import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file

num_classes = 3  # クラス数

# VGG16 モデルを作成する。
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))
vgg16.trainable = False  # 重みをフリーズする。

model = Sequential(
    [
        vgg16,
        Flatten(),
        Dense(500, activation="relu"),
        Dropout(0.5),
        Dense(500, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# コンパイル
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# ハイパーパラメータ
batch_size = 64  # バッチサイズ
num_epochs = 30  # エポック数

# ImageDataGenerator を作成する。
datagen_params = {
    "preprocessing_function": preprocess_input,
    "horizontal_flip": True,
    "brightness_range": (0.7, 1.3),
    "validation_split": 0.2,
}
datagen = image.ImageDataGenerator(**datagen_params)

# dataset_dir
dataset_dir = os.getcwd() + '/imgs/train_images/'
# 学習データを生成するジェネレーターを作成する。
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=model.input_shape[1:3],
    batch_size=batch_size,
    class_mode="sparse",
    subset="training"
)

# バリデーションデータを生成するジェネレーターを作成する。
val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=model.input_shape[1:3],
    batch_size=batch_size,
    class_mode="sparse",
    subset="validation",
)

# クラス ID とクラス名の対応関係
print(train_generator.class_indices)
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}

# 学習する。
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=num_epochs,
)