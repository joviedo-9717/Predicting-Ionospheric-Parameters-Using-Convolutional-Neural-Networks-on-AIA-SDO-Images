# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:35:46 2025

@author: Nutzer
"""

# To develop the Convolutional Neural Network (CNN) algorithm

import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd
from PIL import Image
import os
import datetime
import urllib.request
import re
import numpy as np
import matplotlib.pyplot as plt
OMNI_ROOT = os.environ.get("OMNI_ROOT", "./omni")
omni_lowres_cache = {}



# Step 1: Mount Google Drive

# Step 2: Define the full file path to your CSV in Google Drive
# Note: The path usually starts with '/content/drive/MyDrive/' (no space in MyDrive)
csv_path = 'C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/omni_timeseries1_final.csv'

# Step 3: Read the CSV file using pandas
omni = pd.read_csv(csv_path)

# Optional: Display the first few rows to verify
print(omni.head())

import matplotlib.pyplot as plt

plt.plot(omni.datetime,omni.sunspot)

import os
from PIL import Image


# Set your base directory
base_dir = 'C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/SDO_data/SDO_data/'
wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304']  # Update as needed

def load_and_preprocess_image(img_path, target_size=(512, 512), to_grayscale=False):
    """Load an image, resize, convert to array, and normalize."""
    img = Image.open(img_path)
    if to_grayscale:
        img = img.convert('L')  # Convert to grayscale
    else:
        img = img.convert('RGB')  # Ensure 3 channels

    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    return arr

# Example: Load all images for one wavelength
images = []
image_time = []
f107 = []
dst = []
kp = []
sunspot = []
for wl in wavelengths:
    folder = os.path.join(base_dir, wl)
    i_time = 0
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.jpg'):
            dt = datetime.datetime.strptime(fname[0:15], '%Y%m%d_%H%M%S')
            dt_omni = datetime.datetime.strptime(omni["datetime"][i_time], '%Y-%m-%d %H:%M:%S')
            img_path = os.path.join(folder, fname)
            arr = load_and_preprocess_image(img_path, target_size=(512, 512), to_grayscale=True)
            images.append(arr)
            image_time.append(dt)
            # if statement: to check for time:
            #print(omni["f107"][omni["datetime"] == dt])
            #f10_7.append(omni["f107"])
            if (dt_omni == dt):
                f107.append(omni["f107"][i_time])
                dst.append(omni["dst"][i_time])
                kp.append(omni["kp"][i_time])
                sunspot.append(omni["sunspot"][i_time])
            else:
                print("The world has fallen: " , dt, dt_omni)
                exit(1)
            i_time += 1

# Convert list to numpy array for ML (N, H, W, C)
images_np = np.stack(images)
print(images_np.shape)  # Should be (num_images, 512, 512, 3) for RGB

plt.imshow(images_np[0],cmap="gray")


from sklearn.model_selection import train_test_split

labels = np.stack([f107, kp, dst, sunspot], axis=1)
times = np.stack([image_time,image_time,image_time,image_time],axis=1)
#80% training data, 20% test data 
X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(images_np, labels, times, test_size=0.2, random_state=42)

y_train.shape


from tensorflow.keras import layers, models

from tensorflow.keras.models import load_model
model = load_model('C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/FromFrank/model_4params_100epochs_64_32_32.keras')

y_pred = model.predict(X_test[..., np.newaxis])
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print('Test MSE:', mse)



time_sorted, f107_sorted = zip(*sorted(zip(time_test[:,0],y_test[:,0])))
time_sorted, kp_sorted = zip(*sorted(zip(time_test[:,1],y_test[:,1])))
time_sorted, dst_sorted = zip(*sorted(zip(time_test[:,2],y_test[:,2])))
time_sorted, sunspot_sorted = zip(*sorted(zip(time_test[:,3],y_test[:,3])))


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
ax1.plot(time_sorted,f107_sorted,label=r'$F10.7$',color='r',alpha = 0.6,linewidth=3.5)
ax1.scatter(time_test[:,0],y_pred[:,0],label=r'$F10.7$ (pred)',s=15,color='darkcyan')
ax2.scatter(time_test[:,0],(y_pred[:,0]-y_test[:,0]),label='Difference',s=15,color='purple')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=4)
ax2.set_xlabel("Time",fontsize=24)
ax1.set_ylabel(r'$F10.7$',fontsize=24)
ax2.set_ylabel('Difference',fontsize=24)
ax1.legend(fontsize=19)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
#fig.savefig('C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/Subplot_f107_32_16_16.png')


plt.figure(figsize=(9, 9))
plt.scatter(y_test[:,0],y_pred[:,0],s=20)
plt.plot(y_test[:,0], y_test[:,0], color='r', linewidth=2)
plt.xlabel(r'$F10.7$',fontsize=22)
plt.ylabel(r'$F10.7$ (pred)',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.savefig("C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/ModelComp_f107_32_16_16.png")
plt.show()




fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
ax1.plot(time_sorted,dst_sorted,label=r'$Dst$',color='r',alpha = 0.6,linewidth=3.5)
ax1.scatter(time_test[:,2],y_pred[:,2],label=r'$Dst$ (pred)',s=20,color='darkcyan')
ax2.scatter(time_test[:,2],(y_pred[:,2]-y_test[:,2]),label='Difference',s=15,color='purple')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=4)
ax2.set_xlabel("Time",fontsize=24)
ax1.set_ylabel(r'$Dst$',fontsize=24)
ax2.set_ylabel('Difference',fontsize=24)
ax1.legend(fontsize=19)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
#fig.savefig('C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/Subplot_Dst_bad.png')



plt.figure(figsize=(8, 8))
plt.scatter(y_test[:,2],y_pred[:,2],s=20)
plt.plot(y_test[:,2], y_test[:,2], color='r', linewidth=2)
plt.xlabel(r'$Dst$',fontsize=22)
plt.ylabel(r'$Dst$ (pred)',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.savefig("C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/ModelComp_Dst_bad.png")



fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
ax1.plot(time_sorted,kp_sorted,label=r'10 $Kp$',color='r',alpha = 0.6,linewidth=3.5)
ax1.scatter(time_test[:,1],y_pred[:,1],label=r'10 $Kp$ (pred)',s=20,color='darkcyan')
ax2.scatter(time_test[:,1],(y_pred[:,1]-y_test[:,1]),label='Difference',s=15,color='purple')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=4)
ax2.set_xlabel("Time",fontsize=24)
ax1.set_ylabel(r'10 $Kp$',fontsize=24)
ax2.set_ylabel('Difference',fontsize=24)
ax1.legend(fontsize=19)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
#fig.savefig('C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/Subplot_Kp_bad.png')



plt.figure(figsize=(8, 8))
plt.scatter(y_test[:,1],y_pred[:,1],s=20)
plt.plot(y_test[:,1], y_test[:,1], color='r', linewidth=2)
plt.xlabel(r'10 $Kp$',fontsize=22)
plt.ylabel(r'10 $Kp$ (pred)',fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.savefig("C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/ModelComp_Kp_bad.png")
plt.show()




fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
ax1.plot(time_sorted,sunspot_sorted,label="Sunspot Number",color='r',alpha = 0.6,linewidth=3.5)
ax1.scatter(time_test[:,3],y_pred[:,3],label="Sunspot Number (pred)",s=20,color='darkcyan')
ax2.scatter(time_test[:,3],(y_pred[:,3]-y_test[:,3]),label='Difference',s=15,color='purple')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=4)
ax2.set_xlabel("Time",fontsize=24)
ax1.set_ylabel("Sunspot Number",fontsize=24)
ax2.set_ylabel('Difference',fontsize=24)
ax1.legend(fontsize=19)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='y', labelsize=22)
ax2.tick_params(axis='x', labelsize=22)
#fig.savefig('C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/Subplot_SN_good.png')



plt.figure(figsize=(8, 8))
plt.scatter(y_test[:,3],y_pred[:,3],s=20)
plt.plot(y_test[:,3], y_test[:,3], color='r', linewidth=2)
plt.xlabel("Sunspot Number",fontsize=22)
plt.ylabel("Sunspot Number (pred)",fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.savefig("C:/Users/Nutzer/Documents/ISWC2025/MachineLearning/ModelComp_bad.png")






