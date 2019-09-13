import os
import cv2
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

class_names = list(val_generator.class_indices.keys())

def plot_prediction(img, prediction, label):
    pred_label = np.argmax(prediction)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5), facecolor="w")

    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(
        f"{class_names[pred_label]} {prediction[pred_label]:.2%} ({class_names[label]})",
        fontsize=15,
    )
    
    bar_xs = np.arange(len(class_names))  # 棒の位置
    ax2.bar(bar_xs, prediction)
    ax2.set_xticks(bar_xs)
    ax2.set_xticklabels(class_names, rotation="vertical", fontsize=15)

# def prepare_img(dir_name): 
# ファイル名はとりあえず指定しておく(1.jpg)
# テストデータから3サンプル推論して、結果を表示する。
img_path = '/content/drive/My Drive/Colab Notebooks/imgs/test_images/' + 'lion' + '/79.jpg' # google colaboratoryの場合
label = train_generator.class_indices['lion']
# 画像を読み込む。
img = cv2.imread(img_path)
# 1辺がIMG_SIZEの正方形にリサイズ
img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
# OpenCVの関数cvtColorでBGRとRGBを変換
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# PIL -> numpy 配列
img = np.array(img)
# バッチ次元を追加する。
x = np.expand_dims(img, axis=0)
# 前処理を行う。
x = preprocess_input(x)

# 推論する。
prediction = model.predict(x)

# label, prediction = prepare_img('lion')

# 推論結果を可視化する。
plot_prediction(img, prediction[0], label)
