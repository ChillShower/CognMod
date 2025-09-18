import csv
from tkinter import filedialog
import os
import numpy as np
import sklearn as sk
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

folder = filedialog.askdirectory(title="Select CSV/image Folder")

pcaAnalyze=False
CsvFile=os.path.join(folder,'Final_Grades2.csv')

average=np.array(list(Image.open(os.path.join(folder,"new.jpg")).convert('L').getdata())) 


with open(CsvFile, "r", newline="") as f:
    reader = csv.reader(f)
    data = np.array(list(reader))

print(data[:,0])

imgs = []
valid_images = [".jpg",".gif",".png",".tga"]
grade=[]
i=1 
for f in os.listdir(folder):

    ext = os.path.splitext(f)[1]

    if ext.lower() not in valid_images or f not in data[:,0]:
        continue
    grade.append(data[i][-1])
    i+=1
    imgs.append(np.subtract(np.array(list(Image.open(os.path.join(folder,f)).convert('L').getdata())),average))

width, height = 200,200
imgsArray=np.array(imgs)
print(imgsArray)

pca = PCA(40) # we need 20 principal components.
converted_data = pca.fit_transform(imgsArray)

if pcaAnalyze:
    import matplotlib.pyplot as plt

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

    plt.scatter(converted_data[:, 0], converted_data[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Images")
    plt.show()

    for i in range(1):  # show first 5 components
        
        plt.imshow(np.add(average.reshape((width, height)),max(pca.components_[i])*pca.components_[i].reshape((width, height))), cmap='gray')
        plt.title(f"Principal Component {i+1}, max")
        plt.show()
        plt.imshow(np.add(average.reshape((width, height)),min(pca.components_[i])*pca.components_[i].reshape((width, height))), cmap='gray')
        plt.title(f"Principal Component {i+1}, min")
        plt.show()

    n_components = 40
    imgs_reconstructed = pca.inverse_transform(converted_data)

    plt.figure(figsize=(10,4))
    for i in range(5):
        plt.subplot(2,5,i+1)
        plt.imshow(imgsArray[i].reshape(width, height), cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2,5,i+6)
        plt.imshow(imgs_reconstructed[i].reshape(width, height), cmap='gray')
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()
print(grade)
print(converted_data)
X= converted_data
y=[int(float(grade[i])) for i in range(len(grade))]
print(y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="saga",max_iter=1000)


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

selector = SequentialFeatureSelector(model, direction="forward",cv=5)

selector.fit(X,y)
print('here')
print(selector.get_support())
print(selector.get_params())

import numpy as np

# Get boolean mask of selected features
mask = selector.get_support()

# Reduce X to selected features
X_selected = X[:, mask]

# Refit logistic regression
final_model = LinearRegression()
final_model.fit(X_selected, y)

# Get coefficients
weights = final_model.coef_
print("Weights shape:", weights.shape)
print("Weights:", weights)

selected_features = np.where(mask)[0]  # indices of kept features
analyzeSelection=False
if analyzeSelection:
    selected_features = np.where(mask)[0]  # indices of kept features
    for j in range(4):
        print(f'Weight for grade {j}')
        for i, idx in enumerate(selected_features):
            print(f"Feature {idx}: weight = {weights[j, i]}")
        

    coef_matrix = final_model.coef_  # shape (n_classes=5, n_features_selected)
    avg_abs_coef = np.mean(np.abs(coef_matrix), axis=0)
    top5_idx = np.argsort(avg_abs_coef)[-4:][::-1]  # descending order

    # mask = selector.get_support()
    selected_features = np.where(mask)[0]  # indices of original PCA components that were kept

    # Get the top PCA component indices in original PCA space
    top5_components = selected_features[top5_idx]
    for rank, comp_idx in enumerate(top5_components, 1):
        plt.imshow(pca.components_[comp_idx].reshape((width, height)), cmap='gray')
        plt.title(f"Top {rank}: PCA Component {comp_idx+1}")
        plt.axis("off")
        plt.show()

intercept=final_model.intercept_
coef_matrix = final_model.coef_  # shape (n_classes=5, n_features_selected)

for rating in range(10):


    resultJ0=((rating/2-intercept)/np.linalg.norm(coef_matrix))*coef_matrix
    print(resultJ0)
    weights_pca = np.zeros(pca.n_components_)  # full PCA space
    weights_pca[selected_features] = resultJ0
    print(weights_pca)
    # Back-project into image space
    weights_img = pca.inverse_transform(weights_pca)
    norm_img = (255 * (weights_img - (weights_img+average).min()+average) / ((weights_img+average).ptp())).astype(np.uint8)    
    print(norm_img)
   # Save instead of just showing
    save_path = os.path.join(folder, f"class_{rating/2}_weightmap.jpg")
    Image.fromarray(norm_img.reshape(200,200), mode="L").save(save_path)


