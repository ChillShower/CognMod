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
CsvFile=os.path.join(folder,'Final_Grades.csv')

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
        
        """        plt.imshow(np.add(average.reshape((width, height)),max(pca.components_[i])*pca.components_[i].reshape((width, height))), cmap='gray')
        plt.title(f"Principal Component {i+1}, max")
        plt.show()
        plt.imshow(np.add(average.reshape((width, height)),min(pca.components_[i])*pca.components_[i].reshape((width, height))), cmap='gray')
        plt.title(f"Principal Component {i+1}, min")
        plt.show()"""
        save_path = os.path.join(folder, f"PCA_Comp_min{i}.jpg")

        Image.fromarray(np.add(average.reshape((width, height)),min(pca.components_[i])*pca.components_[i].reshape((width, height))), mode="L").save(save_path)
        save_path = os.path.join(folder, f"PCA_Comp_max{i}.jpg")
        Image.fromarray(np.add(average.reshape((width, height)),max(pca.components_[i])*pca.components_[i].reshape((width, height))), mode="L").save(save_path)




    n_components = 20
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
y=[float(grade[i]) for i in range(len(grade))]
print(y)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

model = LinearRegression()


from sklearn.feature_selection import SequentialFeatureSelector

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
analyzeSelection=True
if analyzeSelection:
    selected_features = np.where(mask)[0]  # indices of kept features
    """    for j in range(4):
        print(f'Weight for grade {j}')
        for i, idx in enumerate(selected_features):
            print(f"Feature {idx}: weight = {weights[j, i]}")"""
        

    coef_matrix = final_model.coef_  # shape (n_classes=5, n_features_selected)
    avg_abs_coef = np.abs(coef_matrix)   # directly absolute values, not mean
    top5_idx = np.argsort(avg_abs_coef)[-5:][::-1]  # descending order
    topk = min(5, len(avg_abs_coef))
    top5_idx = np.argsort(avg_abs_coef)[-topk:][::-1]
    top5_components = selected_features[top5_idx]
    # mask = selector.get_support()


    print("selected_features:", selected_features)
    print("len(selected_features):", len(selected_features))
    print("top5_idx:", top5_idx)
    print("top5_components:", top5_components)
    oldcomp=average.reshape((200,200))
    for rank, comp_idx in enumerate(top5_components, 1):
        """        plt.imshow(pca.components_[comp_idx].reshape((width, height)), cmap='gray')
        plt.title(f"Top {rank}: PCA Component {comp_idx+1}")
        plt.axis("off")
        plt.show()"""
        """        save_path = os.path.join(folder, f"PCA_Comp_min{comp_idx}.jpg")

        comp = pca.components_[comp_idx].reshape((width, height))
        avg = average.reshape((width, height))

        # construct the image you showed with plt (use min or max as you prefer)
        img = avg + np.min(comp) * comp   # same as np.add(...)

        # guard against NaNs/Infs
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize to 0..255 and convert to uint8
        img_min = img.min()
        img_ptp = img.ptp()  # max-min
        if img_ptp == 0:
            img_uint8 = np.clip(img + img_min, 0, 255).astype(np.uint8)
        else:
            img_uint8 = ((img + img_min) / img_ptp * 255.0).astype(np.uint8)


        Image.fromarray(img_uint8, mode='L').save(save_path)
        save_path = os.path.join(folder, f"PCA_Comp_max{comp_idx}.jpg")

        comp = pca.components_[comp_idx].reshape((width, height))
        avg = average.reshape((width, height))

        # construct the image you showed with plt (use min or max as you prefer)
        img = avg + np.max(comp) * comp   # same as np.add(...)

        # guard against NaNs/Infs
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize to 0..255 and convert to uint8
        img_min = img.max()
        img_ptp = img.ptp()  # max-min
        if img_ptp == 0:
            img_uint8 = np.clip(img + img_min, 0, 255).astype(np.uint8)
        else:
            img_uint8 = ((img + img_min) / img_ptp * 255.0).astype(np.uint8)


        Image.fromarray(img_uint8, mode='L').save(save_path)"""
        """        avg = average.reshape((width, height))
        scores = converted_data[:, comp_idx]   # all projected values along component k
        minscore=scores.min()
        maxscore=scores.max()

        print(comp_idx)
        comp = pca.components_[comp_idx].reshape((width, height))
        print(comp)
        print(avg)
        oldcomp=comp
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))  # 1 row, 3 columns
        print(np.linalg.norm(avg))
        print(np.linalg.norm(comp.max()*comp))
        # First image
        axes[0].imshow(avg+minscore*comp, cmap="gray")
        axes[0].set_title("min")
        axes[0].axis("off")

        # Second image
        axes[1].imshow(avg, cmap="gray")
        axes[1].set_title("average")
        axes[1].axis("off")

        #
        axes[2].imshow(avg+maxscore*comp, cmap="gray")  # red/blue shows differences
        axes[2].set_title("max")
        axes[2].axis("off")
        
        axes[3].imshow(avg+maxscore*comp*200-minscore*comp*200, cmap="seismic")  # red/blue shows differences
        axes[2].set_title("difference")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()"""

        scores = converted_data[:, comp_idx]   # all projected values along component k
        pc=pca.components_[comp_idx].reshape((height, width))
        avg_img = average.reshape((height, width))
        minscore=scores.min()
        maxscore=scores.max()
        img_min = avg_img + minscore * pc
        img_avg = avg_img
        img_max = avg_img + maxscore * pc

        # Normalize for display (0..255)
        def to_uint8(img):
            img = np.nan_to_num(img)  # avoid NaN/inf
            img_min, img_ptp = img.min(), img.ptp()
            if img_ptp == 0:
                return np.clip(img - img_min, 0, 255).astype(np.uint8)
            return ((img - img_min) / img_ptp * 255.0).astype(np.uint8)

        img_min = to_uint8(img_min)
        img_avg = to_uint8(img_avg)
        img_max = to_uint8(img_max)

        # Plot side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_min, cmap="gray")
        axes[0].set_title(f"PC{comp_idx+1} - min")
        axes[0].axis("off")

        axes[1].imshow(img_avg, cmap="gray")
        axes[1].set_title("Average image")
        axes[1].axis("off")

        axes[2].imshow(img_max, cmap="gray")
        axes[2].set_title(f"PC{comp_idx+1} - max")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


intercept=final_model.intercept_
coef_matrix = final_model.coef_  # shape (n_classes=5, n_features_selected)
oldnorm_img=0
oldResultJ0=0
"""for rating in range(10):


resultJ0=((rating/2-intercept)/np.linalg.norm(coef_matrix))*coef_matrix
print(resultJ0==oldResultJ0)
oldResultJ0=resultJ0
weights_pca = np.zeros(pca.n_components_)  # full PCA space
weights_pca[selected_features] = resultJ0

print(weights_pca)
# Back-project into image space
weights_img = pca.inverse_transform(weights_pca)
plt.imshow((weights_img).reshape(200,200),cmap='grey')
plt.show()"""
"""denom = (weights_img + average).ptp()
if denom == 0:
    denom = 1
norm_img = (255 * (weights_img + average - (weights_img + average).min()) / denom).astype(np.uint8)"""

"""    print(norm_img)
print(np.array_equal(oldnorm_img, norm_img))

oldnorm_img=norm_img

# Save instead of just showing
save_path = os.path.join(folder, f"class_{rating/2}_weightmap.jpg")
Image.fromarray(norm_img.reshape(200,200), mode="L").save(save_path)"""
ratings = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5.5]

coef = final_model.coef_      # (n_features_selected,)
intercept = final_model.intercept_

synthetic_images = []

for r in ratings:
    # scale along the direction of coef
    z = ((r - intercept) / np.dot(coef, coef)) * coef
    
    # put back into full PCA space
    weights_pca = np.zeros(pca.n_components_)
    weights_pca[selected_features] = z
    
    # reconstruct
    img = pca.inverse_transform(weights_pca) + average
    img_uint8 = ((img - img.min()) / img.ptp() * 255).astype(np.uint8)
    synthetic_images.append(img_uint8.reshape((height, width)))
    save_path = os.path.join(folder, f"class_{r}_weightmap.jpg")
    Image.fromarray(img_uint8.reshape(200,200), mode="L").save(save_path)

fig, axes = plt.subplots(1, len(ratings), figsize=(20, 4))
for ax, img, r in zip(axes, synthetic_images, ratings):
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{r}")
    ax.axis("off")
plt.show()



