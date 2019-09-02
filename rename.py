import os

def rename_files(file_name):
  files = os.listdir(os.getcwd() + '/imgs/train_images/' + file_name +'/')
  for i, old_name in enumerate(files):
    number = old_name.split('.')[0]
    new_name = number + '.jpg'
    os.rename(os.getcwd() + '/imgs/train_images/' + file_name +'/' + old_name, os.getcwd() + '/imgs/train_images/' + file_name +'/' + new_name)
    print(old_name + '→' + new_name)
rename_files('イカ')
