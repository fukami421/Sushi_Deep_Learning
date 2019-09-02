import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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
# def make_train_data():
for i, dir_name in enumerate(img_dirs):
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
                # resize後の画像サイズを取得
                height, width, ch = img.shape
                img_column_rgb = []# 列
                for h in range(height):
                        img_row_rgb = []# 行
                        for w in range(width):
                                img_cell_rgb = img[h, w]
                                img_row_rgb.append(img_cell_rgb)
                        img_column_rgb.append(img_row_rgb)
                train_images.append(img_column_rgb)

                # ラベル
                tmp = np.zeros(NUM_CLASSES)#0で初期化されたndarrayを生成する関数
                tmp[i] = i
                train_labels.append(tmp[i])

# numpy配列に変換
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

#テスト用データセット作成
# def make_test_data():
for i, dir_name in enumerate(img_dirs):
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
                # resize後の画像サイズを取得
                height, width, ch = img.shape
                img_column_rgb = []# 列
                for h in range(height):
                        img_row_rgb = []# 行
                        for w in range(width):
                                img_cell_rgb = img[h, w]
                                img_row_rgb.append(img_cell_rgb)
                        img_column_rgb.append(img_row_rgb)
                test_images.append(img_column_rgb)

                # ラベル
                tmp = np.zeros(NUM_CLASSES)#0で初期化されたndarrayを生成する関数
                tmp[i] = i
                test_labels.append(tmp[i])

# numpy配列に変換
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
    
# make_train_data()
# make_test_data()

class_names = ['Squid', 'Octopus', 'Tuna',]

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)
# plt.show()

#ニューラルネットワークにデータを投入する前に、これらの値を0から1までの範囲にスケールする
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fit = model.fit(
        train_images, train_labels,
        batch_size=128,
        epochs=50,
        verbose=2,
        # validation_data=(X_valid, Y_valid),
        callbacks=[])              

# model.fit(train_images, train_labels, epochs=10)
# plot_model(model, to_file='./model.png')
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print('predictions[0]: ', predictions[0])
print('一番信頼度が高いラベル: ', np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(img, cmap=plt.cm.binary)
        
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        
        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[int(true_label)]),
                                        color=color)
    
def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="loss for training")
    axR.plot(fit.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig('./mnist-tutorial.png')
plt.close()
