import cv2
import os

IMG_SIZE = 224

def resize(dirname1, dirname2):
		dir_name = dirname2
		files = os.listdir(os.getcwd() + '/imgs/' + dirname1 + '/train_images/' + dirname2)
		save_dir = os.getcwd() + '/imgs/' + dirname1 + '/train_images/' + dirname2 + '/'

		for file in files:
				# 画像読み込み
				img = cv2.imread(os.getcwd() + '/imgs/' + dirname1 + '/train_images/' + dirname2 + '/' + file)
				if img is not None:                
								# 1辺がIMG_SIZEの正方形にリサイズ
								img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
								cv2.imwrite(save_dir + file, img)

resize("sushi", "amaebi")