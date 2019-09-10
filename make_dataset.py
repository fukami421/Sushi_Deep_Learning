import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

NUM_CLASSES = 3 # 分類するクラス数
IMG_SIZE = 280 # 画像の1辺の長さ

# 画像のあるディレクトリ
img_dirs = ['dog', 'lion', 'cat']

# class name
class_names = ['dog', 'lion', 'cat',]

# 学習用画像データ
train_images = []
# 学習用データのラベル
train_labels = []
# テスト用画像データ
test_images = []
# テスト用データのラベル
test_labels = []

#学習用データセット作成
for label, dir_name in enumerate(img_dirs):
    # ./data/以下の各ディレクトリ内のファイル名取得
    files = os.listdir(os.getcwd() + '/imgs/train_images/' + dir_name)
    for file in files:
        # 画像読み込み
        img = cv2.imread(os.getcwd() + '/imgs/train_images/' + dir_name + '/' + file)
        if img is not None:                
                # 1辺がIMG_SIZEの正方形にリサイズ
                img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                # OpenCVの関数cvtColorでBGRとRGBを変換
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Numpy配列にする
                data = np.asarray(img)
                # 配列に追加
                train_images.append(data)
                # ラベル
                train_labels.append(label)

# numpy配列に変換
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#テスト用データセット作成
for label, dir_name in enumerate(img_dirs):
    # ./data/以下の各ディレクトリ内のファイル名取得
    files = os.listdir(os.getcwd() + '/imgs/test_images/' + dir_name)
    for file in files:
        # 画像読み込み
        img = cv2.imread(os.getcwd() + '/imgs/test_images/' + dir_name + '/' + file)
        if img is not None:
                # 1辺がIMG_SIZEの正方形にリサイズ
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # OpenCVの関数cvtColorでBGRとRGBを変換
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Numpy配列にする
                data = np.asarray(img)
                # 配列に追加
                test_images.append(data)
                # ラベル
                test_labels.append(label)

# numpy配列に変換
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# ニューラルネットワークにデータを投入する前に、これらの値を0から1までの範囲にスケールする
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-Hotエンコーディングする
train_labels = np_utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = np_utils.to_categorical(test_labels, NUM_CLASSES)

# モデルの構築
model = models.Sequential()
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(IMG_SIZE, IMG_SIZE, NUM_CLASSES)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128,(3,3),activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128,(3,3),activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES,activation="softmax"))

#モデルのコンパイル
model.compile(optimizer=optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0),
              loss="categorical_crossentropy",
              metrics=["acc"])

# Early-stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

# モデルの学習
fit = model.fit(
        train_images, train_labels,
        batch_size=10,
        epochs=10,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stopping])

# モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

plot_history(fit)

# 予測する
predictions = model.predict(test_images, batch_size=6)
print('一番信頼度が高いラベル: ', np.argmax(predictions[0]))
print('正解は: ', test_labels[0])
print('一番信頼度が高いラベル: ', np.argmax(predictions[1]))
print('正解は: ', test_labels[1])
print('一番信頼度が高いラベル: ', np.argmax(predictions[2                 ]))
print('正解は: ', test_labels[2])
print('一番信頼度が高いラベル: ', np.argmax(predictions[3]))
print('正解は: ', test_labels[3])

# plt.figure()
# plt.imshow(test_images[0])
# plt.show()

# plt.figure()
# plt.imshow(test_images[1])
# plt.show()

# モデル保存
# model.save('model.h5', include_optimizer=False)
