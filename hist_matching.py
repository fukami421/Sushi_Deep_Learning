'''OpenCVを使ったhist matching'''

import os
import cv2
from natsort import natsorted
from matplotlib import pyplot as plt

TARGET_FILE = '0.jpg'
IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/大トロ/'
# IMG_SIZE = (200, 200)

#比較される側のヒストグラムを用意
target_img_path = IMG_DIR + TARGET_FILE
target_img = cv2.imread(target_img_path)
# target_img = cv2.resize(target_img, IMG_SIZE)
target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])
print('TARGET_FILE: %s' % (TARGET_FILE))

#比較する側のファイル
files = os.listdir(IMG_DIR)

#ファイルを名前順に並び替え
sorted_files = []
for file in natsorted(files):
    sorted_files.append(file)
print(sorted_files)

#比較する側のヒストグラムを用意
x = list(range(len(sorted_files)))
x.remove(0)
y = []
for file in sorted_files:
    if file == '.DS_Store' or file == TARGET_FILE:
        continue
    comparing_img_path = IMG_DIR + file
    comparing_img = cv2.imread(comparing_img_path)
    # comparing_img = cv2.resize(comparing_img, IMG_SIZE)
    comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

    ret = cv2.compareHist(target_hist, comparing_hist, 0)
    print('[' + file + ']'+ 'の一致率は ', ret)
    y.append(ret)

#グラフをplotする
plt.figure()
plt.plot(x,y)
plt.title('Matching rate with 1.jpg')
plt.xlabel('Image label')
plt.ylabel('Matching rate')
plt.savefig('./result.png')
plt.show()
