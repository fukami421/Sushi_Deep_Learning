# Sushi_Deep_Learning
機械学習を用いて、お寿司のネタを判別しようというものです。

## 用いたもの
### TensorFlow
https://www.tensorflow.org/tutorials/keras/basic_classification 
機械学習モデルの開発およびトレーニングに役立つオープンソースのコア ライブラリです。
#### Setup
```command line
$ pip install tensorflow
```

### google_images_download
#### Setup
```command line
# install package
$ pip install google_images_download

# install chromedriver
http://chromedriver.chromium.org/downloads
```

## Execution command
```command line
# google_images_downloadで画像をダウンロード
$ googleimagesdownload --keywords "指定したい検索ワード" --limit 1000 --output_directory "./imgs/test_images" --chromedriver './../chromedriver' --format 'jpg';
指定したい検索ワードで検索した結果、得られたjpgファイルを1000個 .imgs/test_images/指定したい検索ワード/ にダウンロードします。

# 指定されたディレクトリ以下にあるファイルの名前を変更
$ python3 rename.py 

# 学習させて、識別結果の正確性を数値化
$ python3 make_dataset.py
``` 
