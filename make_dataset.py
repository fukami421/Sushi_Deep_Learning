import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.utils import np_utils

NUM_CLASSES = 3 # 分類するクラス数
IMG_SIZE = 280 # 画像の1辺の長さ

# 画像のあるディレクトリ
img_dirs = ['イカ', 'タコ', 'マグロ']

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

#ニューラルネットワークにデータを投入する前に、これらの値を0から1までの範囲にスケールする
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

class_names = ['Squid', 'Octopus', 'Tuna',]

# train_labels = np_utils.to_categorical(train_labels, NUM_CLASSES)
# test_labels = np_utils.to_categorical(test_labels, NUM_CLASSES)
# train_labels = np.ndarray.flatten(train_labels)
# test_labels = np.ndarray.flatten(test_labels)
#モデルの構築
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(3,activation="sigmoid")) #分類先の種類分設定

#モデルのコンパイル
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])

fit = model.fit(
        train_images, train_labels,
        batch_size=6,
        nb_epoch=10,)
        # verbose=2,)
        # validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
def plot_history(history):
    # print(history.history.keys())

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
# predictions = model.predict(test_images)
# print('predictions[0]: ', predictions[0])
# print('一番信頼度が高いラベル: ', np.argmax(predictions[0]))

# def plot_image(i, predictions_array, true_label, img):
#         predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#         plt.grid(False)
#         plt.xticks([])
#         plt.yticks([])
        
#         plt.imshow(img, cmap=plt.cm.binary)
        
#         predicted_label = np.argmax(predictions_array)
#         if predicted_label == true_label:
#             color = 'blue'
#         else:
#             color = 'red'
        
#         plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                         100*np.max(predictions_array),
#                                         class_names[int(true_label)]),
#                                         color=color)
    
# def plot_value_array(i, predictions_array, true_label):
#         predictions_array, true_label = predictions_array[i], true_label[i]
#         plt.grid(False)
#         plt.xticks([])
#         plt.yticks([])
#         thisplot = plt.bar(range(10), predictions_array, color="#777777")
#         plt.ylim([0, 1])
#         predicted_label = np.argmax(predictions_array)

#         thisplot[predicted_label].set_color('red')
#         thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# # loss
# def plot_history_loss(fit):
#     # Plot the loss in the history
#     axL.plot(fit.history['loss'],label="loss for training")
#     axL.plot(fit.history['val_loss'],label="loss for validation")
#     axL.set_title('model loss')
#     axL.set_xlabel('epoch')
#     axL.set_ylabel('loss')
#     axL.legend(loc='upper right')

# # acc
# def plot_history_acc(fit):
#     # Plot the loss in the history
#     axR.plot(fit.history['acc'],label="loss for training")
#     axR.plot(fit.history['val_acc'],label="loss for validation")
#     axR.set_title('model accuracy')
#     axR.set_xlabel('epoch')
#     axR.set_ylabel('accuracy')
#     axR.legend(loc='upper right')

# plot_history_loss(fit)
# plot_history_acc(fit)
# fig.savefig('./mnist-tutorial.png')
# plt.close()
