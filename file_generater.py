import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

""" ImageDataGeneratorのテンプレ
ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=90.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0)
"""

# 指定したディレクトリ内のファイルを取得
files = os.listdir(os.getcwd() + '/imgs/train_images/' + 'イカ')
 
# 出力ディレクトリを作成
output_dir = os.getcwd() + '/imgs/train_images/イカ2/'
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

# 画像を変換して新たに作成  
for i, file in enumerate(files):
    # if os.path.isdir(file) is True:
        if file == '.DS_Store':
            continue

        img = load_img(os.getcwd() + '/imgs/train_images/' + 'イカ/' + file)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        # ImageDataGeneratorの生成
        datagen = ImageDataGenerator(
          rotation_range=90.0,
        )
    
        # 9個の画像を生成します
        g = datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='', save_format='jpg')
        for i in range(9):
            batch = g.next()