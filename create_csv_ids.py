import os
import pandas as pd
import shutil

file_dir = "/home/ruben/Documentos/dataset/train/masks/"
list_ids = []

for root, _, files in os.walk(file_dir):
    pass

for file in files:
    ext = file[-4:]
    if ext == ".png":
        index = file[:len(file)-17] # tira o '_segmentation.png'
        shutil.copy("/home/ruben/Documentos/dataset/train/masks/" + file, "/home/ruben/Documentos/dataset/train/masks1/"+index+'.png')

#df = pd.DataFrame(list_ids, columns=['id'])
#print(df)
#df.to_csv("/home/ruben/Documents/dataset/train.csv")