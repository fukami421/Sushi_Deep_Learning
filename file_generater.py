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

def get_files(dir_name):
    # 指定したディレクトリ内のファイルを取得
    files = os.listdir(os.getcwd() + '/imgs/' + dir_name)
    return files

def make_dir(new_dir_name):
    # 出力ディレクトリを作成
    output_dir = os.getcwd() + '/imgs/' + new_dir_name + '/'
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    return output_dir

def make_img_file(dir_name, n):
    # 画像を変換して新たに作成  
    output_dir = make_dir(dir_name)
    for i, file in enumerate(get_files(dir_name)):
        if file == '.DS_Store':
            continue

        img = load_img(os.getcwd() + '/imgs/' + dir_name + '/' + file)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        # ImageDataGeneratorの生成(ここを変更するとそれに沿った画像が生成される)
        datagen = ImageDataGenerator(
          rotation_range=10.0,
          horizontal_flip=True,
          width_shift_range=0.01,
          height_shift_range=0.01,
          shear_range=0.01,
        #   zoom_range=0.1,
          fill_mode='nearest',
        )
    
        # n個の画像を生成します
        created_img = datagen.flow(x, batch_size=1, save_to_dir = output_dir, save_prefix = '', save_format = 'jpg')
        for i in range(n):
            batch = created_img.next()

make_img_file('sushi/train_images/kohada', 15)