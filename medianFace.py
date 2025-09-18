import csv
from tkinter import filedialog
import os
import numpy as np
from PIL import Image

folder = filedialog.askdirectory(title="Select CSV/image Folder")


CsvFile=os.path.join(folder,'Final_Grades2.csv')

imgs = []
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(folder):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(folder,f)).convert('L'))

width, height = 200,200
print(imgs)

average=np.array(list(imgs[0].getdata()))
for elt in imgs:
    pixels= np.array(list(elt.getdata()))
    average=np.add(average,pixels)


average=average*(1/len(imgs))

averageFinal=np.array([[average[i+j*200] for i in range(200)]for j in range(200)])
new_image = Image.fromarray(averageFinal)

if new_image.mode != 'RGB':
    new_p = new_image.convert('RGB')
new_p.save('new.jpg')