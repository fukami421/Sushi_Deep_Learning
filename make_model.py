import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file

num_classes = 12  # クラス数

# VGG16 モデルを作成する。
vgg16 = VGG16(include_top=False, weights = "imagenet",  input_tensor = Input(shape=(224, 224, 3)))

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
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# ハイパーパラメータ
batch_size = 64  # バッチサイズ
num_epochs = 35  # エポック数

# ImageDataGenerator を作成する。
datagen_params = {
    "preprocessing_function": preprocess_input,
    "horizontal_flip": True,
    "brightness_range": (0.7, 1.3),
    "validation_split": 0.2,
}

datagen = ImageDataGenerator(**datagen_params)
# dataset_dir
dataset_dir = '/content/drive/My Drive/Colab Notebooks/imgs/sushi/train/'
validation_dir = '/content/drive/My Drive/Colab Notebooks/imgs/sushi/validation/'

train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
# 学習データを生成するジェネレーターを作成する。
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

# バリデーションデータを生成するジェネレーターを作成する。
val_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

# クラス ID とクラス名の対応関係
print(train_generator.class_indices)

# 学習する。
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=num_epochs,
)

epochs = np.arange(1, num_epochs + 1)
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))

# 損失関数の履歴を可視化する。
ax1.plot(epochs, history.history["loss"], label="loss")
ax1.plot(epochs, history.history["val_loss"], label="validation loss")
ax1.set_xlabel("epochs")
ax1.legend()

# 精度の履歴を可視化する。       
ax2.plot(epochs, history.history["acc"], label="accuracy")
ax2.plot(epochs, history.history["val_acc"], label="validation accuracy")
ax2.set_xlabel("epochs")
ax2.legend()

plt.show()

# 評価する。
test_loss, test_acc = model.evaluate_generator(val_generator)
print(f"test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}")
