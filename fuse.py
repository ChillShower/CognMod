import csv
from tkinter import filedialog
import os
import numpy as np

folder = filedialog.askdirectory(title="Select CSV Folder")
nameCsv1="first_review.csv"
nameCsv2="second_review.csv"

ResultOne= os.path.join(folder, nameCsv1)
ResultTwo= os.path.join(folder, nameCsv2)

NewCsv=os.path.join(folder,'Result_Alex.csv')

with open(NewCsv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "grade1","grade2"])


with open(ResultOne, "r", newline="") as f:
        reader = csv.reader(f)
        data1 = list(reader)
print(data1)



with open(ResultTwo, "r", newline="") as f:
        reader = csv.reader(f)
        data2 = list(reader)
listFinal=[]
"""transposed_data2Test = zip(*data2)
transposed_data2Test.sort(key= lambda x: x[0],reversed=True)
data2Test=zip(*transposed_data2Test)"""
data2.sort()
data1.sort()
print(np.sum( [data2[i][0]== data1[i][0] for i in range(min(len(data2),len(data1)))]))

for i in range(len(data1)):
    elt=data1[i]
    for j in range(len(data2)):
        if elt[0]== data2[j][0]:
            if elt[1]=="keep" and data2[j][1] =="keep":
                listFinal.append([elt[0],elt[2],data2[j][2]])


with open(NewCsv, "a", newline="") as f:
    writer = csv.writer(f)
    for elt in listFinal:
        writer.writerow(elt)