from google_images_download import google_images_download  #モジュールのインポート
import os

def google_images_download(keyword):
    response = google_images_download.googleimagesdownload()  #responseオブジェクトの生成
    arguments = {"keywords":keyword,  # 検索キーワード
                "limit":1000,  # ダウンロードする画像の数(デフォルト100)
                #"no_numbering":True,  #ナンバリングを無しにする
                "output_directory": "./imgs/test_images/",
                "chromedriver": "./../chromedriver",
                "format": "jpg",
                }
    response.download(arguments)   # argumentsをresponseオブジェクトのdownloadメソッドに渡す

google_images_download('マダコ属')

'''そのまま打ち込む場合'''
#$ googleimagesdownload --keywords "マダコ属" --limit 1000 --output_directory "./imgs/test_images" --chromedriver './../chromedriver' --format 'jpg';
