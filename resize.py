import cv2
import os

IMG_SIZE = 224
dir_name = "cat"
files = os.listdir(os.getcwd() + '/animal_imgs/train/' + dir_name)
save_dir = os.getcwd() + '/animal_imgs/train/' + dir_name + '/'

for file in files:
    # 画像読み込み
    img = cv2.imread(os.getcwd() + '/animal_imgs/train/' + dir_name + '/' + file)
    if img is not None:                
        # 1辺がIMG_SIZEの正方形にリサイズ
        img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_dir + file, img)
