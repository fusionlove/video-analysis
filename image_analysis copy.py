# Version 0.1 Jan 17 2020


import scipy
import pickle, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from skimage.util import img_as_float
from sklearn.model_selection import train_test_split
from skimage.color import rgb2hsv



import skimage
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.io import imsave



colours = []

#Cold colours

# colours.append([1, 176, 0])
# colours.append([88, 192, 23])
# colours.append([0, 214, 245])
# colours.append([ 10,  144,  59])
# colours.append([0, 75, 255])
# colours.append([56, 8, 122])

# #Warm colours

# colours.append([224, 215, 26])
# colours.append([245, 0, 2])
# colours.append([255, 0, 136])
# colours.append([ 112, 29, 0])
# colours.append([ 191, 33, 3])
# colours.append([ 166, 12, 96])

# Colours from screenshots:

colours.append([  0, 168,  31]) # C1.png
colours.append([ 63, 186,  42]) # C2.png
colours.append([  0, 208, 241]) # C3.png
colours.append([  0, 134,  56]) # C4.png
colours.append([ 23,  57, 249]) # C5.png
colours.append([ 52,   0, 107]) # C6.png

colours.append([219, 210,  52]) # W1.png
colours.append([251,   0,  27]) # W2.png
colours.append([255,   0, 125]) # W3.png
colours.append([104,  27,  10]) # W4.png
colours.append([255,  79,  33]) # W5.png
colours.append([161,   6,  85]) # W6.png



colours
len(colours)

for col in colours:
  for i in range(3):
    col[i] = col[i]/255.0


def convert(im):
  im = im[...,::-1]
  return img_as_float(im)

def show(image):
  fig,ax = plt.subplots()
  ax.imshow(image)
  # convert from BGR to RGB

def dump_frames(fv,folder):
  # Folder has to have slash on the end
  import cv2
  vidcap = cv2.VideoCapture(fv)


  # Dump the entire video to a folder
  count = 0
  success = True
  while success:
    success,image = vidcap.read()
    if image is None:
      break
    image = convert(image)
    image = skimage.img_as_ubyte(image)
    print(count)
    if success:
      imsave(folder+f"frame{count:04}.jpg" , image)     # save frame as JPEG file

    count += 1

def place_square(im, R, G, B):
  # Specify image, patch colour values
  # in 0 - 1 RGB space
  si,sj,sc = im.shape
  y = int(si/2)
  x = int(sj/2)

  r = 40
  n = 80
  

  # Set square to square colour
  im[y-r:y+r,x-r:x+r,:] = [R,G,B]
  # Make a copy and blank out the square
  im_copy = im.copy()
  im_copy[y-r:y+r,x-r:x+r,:] = np.nan
  
  # Extract the patch of interest (NaN square and surround)
  
  patch = im_copy[y-n:y+n,x-n:x+n,:] 

  print('patch value:')

  print(patch)
  
  plt.imshow(patch)
  plt.show()
  
  # Convert patch to HSV
  dbpatch = patch
  patch_hsv = rgb2hsv(patch)
 # dıfferent here
  
  h1 = np.mean(patch_hsv[:,:,0])
  s1 = np.mean(patch_hsv[:,:,1])
  v1 = np.mean(patch_hsv[:,:,2])
  
  print(f"Patch HSV {round(h1,2)} {round(s1,2)} {round(v1,2)}")
  
  # Generate a small test square and set colours to square colour
  square = np.zeros((10,10,3))
  square[:,:,0] = R
  square[:,:,1] = G
  square[:,:,2] = B
  square_hsv = rgb2hsv(square)

  h2 = np.mean(square_hsv[:,:,0])
  s2 = np.mean(square_hsv[:,:,1])
  v2 = np.mean(square_hsv[:,:,2])
  
  print(f"Square HSV {round(h2,2)} {round(s2,2)} {round(v2,2)}")

  # Differences
  dh = h1 - h2
  ds = s1 - s2
  dv = v1 - v2

  dffs = [dh,ds,dv]
  
  # Euclidean distance in 3D HSV space
  hsv_distance = np.sqrt( (h1-h2)**2 + (s1-s2)**2 + (v1-v2)**2)
  return im, patch, hsv_distance, dffs, dbpatch



def run_analysis(folder, initial_delay, delay):# Loop over the images, loading them and doing the analysis
# With float images so can use NaN
# Also write out to file

  hsv_distances = []



  # Loop over the images, loading them and doing the analysis


  go = True

  trial_index = 0
  while go:
    seconds = initial_delay + (trial_index * delay)
    trial_number = trial_index + 1
    frame_number = seconds * 30 # fps = 30
    print(f"trial {trial_number}: frame {frame_number} at {seconds} seconds")
    filename = folder + f"frame{frame_number:04}.jpg"
    print(filename)
    image = imread(filename)
    
    image = skimage.img_as_float(image)
    #show(image)
    dbg_image = image
    

    cr = colours[trial_index][0]
    cg = colours[trial_index][1]
    cb = colours[trial_index][2]
    im2, patch,d, dffs, dbpatch = place_square(image,cr,cg,cb)

    show(im2)
    show(patch)
    print(f"HSV distance = {d}")
    hsv_distances.append(d)
    im2 = skimage.img_as_ubyte(im2)
    imsave(folder+f"trial_{trial_number}_frame_{frame_number:04}_s_{seconds}.png", im2)

    trial_index += 1
    if trial_index > 0:
      go = False
      
  return hsv_distances, dbg_image, dffs, dbpatch
