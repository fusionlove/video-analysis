# Version 0.1 Jan 17 2020


import scipy
import pickle, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from skimage.util import img_as_float
from sklearn.model_selection import train_test_split
from skimage.color import rgb2hsv, hsv2rgb



import skimage
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.io import imsave
import colorsys
from os.path import exists

MAX_TRIALS = 10

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

COLOURS = {}

COLOURS['Cold1'] =[  0, 168,  31]
COLOURS['Cold2'] =[ 63, 186,  42]
COLOURS['Cold3'] =[  0, 208, 241]
COLOURS['Cold4'] =[  0, 134,  56]
COLOURS['Cold5'] =[ 23,  57, 249]
COLOURS['Cold6'] =[ 52,   0, 107]

COLOURS['Complex1'] =[129.7490654,   97.34760666,  93.3445015 ] 
COLOURS['Complex2'] =[86.40116882, 64.69316864, 59.77134323] 
COLOURS['Complex3'] =[76.36225128, 60.87407303, 51.75866699] 
COLOURS['Complex4'] =[79.48569489, 73.43889999, 37.7695961 ] 
COLOURS['Complex5'] =[48.50060018, 43.69769796, 42.55472565] 
COLOURS['Complex6'] =[84.87594223, 84.2996521,  66.03577805] 

COLOURS['Simple1'] =[155.58521271,  92.51436996,  59.08986664] 
COLOURS['Simple2'] =[160.67521286, 112.43297195,  70.450634  ] 
COLOURS['Simple3'] =[115.07292557, 182.62963486,  53.78518295] 
COLOURS['Simple4'] =[185.91750717,  83.67290115, 187.16868973] 
COLOURS['Simple5'] =[175.52278519, 175.51542664,  27.80386353] 
COLOURS['Simple6'] =[210.55834961, 155.34825897,  75.88077927] 

COLOURS['Warm1'] = [219, 210,  52]
COLOURS['Warm2'] =[251,   0,  27]
COLOURS['Warm3'] =[255,   0, 125]
COLOURS['Warm4'] =[104,  27,  10]
COLOURS['Warm5'] =[255,  79,  33]
COLOURS['Warm6'] =[161,   6,  85]


#  [129.7490654   97.34760666  93.3445015 ] 
# [86.40116882 64.69316864 59.77134323] 
# [76.36225128 60.87407303 51.75866699] 
# [79.48569489 73.43889999 37.7695961 ] 
# [48.50060018 43.69769796 42.55472565] 
# [84.87594223 84.2996521  66.03577805] 
# [155.58521271  92.51436996  59.08986664] 
# [160.67521286 112.43297195  70.450634  ] 
# [115.07292557 182.62963486  53.78518295] 
# [185.91750717  83.67290115 187.16868973] 
# [175.52278519 175.51542664  27.80386353] 
# [210.55834961 155.34825897  75.88077927] 



colours
len(colours)

for col in colours:
  for i in range(3):
    col[i] = col[i]/255.0
    
def rgb2hsv_2(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def convert_rgb_hsv(x):
  #colorsys.rgb_to_hsv(x)
  #return rgb2hsv_2(x)
  return rgb2hsv(x)
    
    
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

  # print('patch value:')

  # print(patch)
  
  # plt.imshow(patch)
  # plt.show()
  
  # Convert patch to HSV
  #patch_hsv = rgb2hsv(patch)
  patch_hsv = convert_rgb_hsv(patch)
  
  h1 = np.mean(patch_hsv[:,:,0])
  s1 = np.mean(patch_hsv[:,:,1])
  v1 = np.mean(patch_hsv[:,:,2])
  
  # print(f"Patch HSV {round(h1,2)} {round(s1,2)} {round(v1,2)}")
  
  # Generate a small test square and set colours to square colour
  square = np.zeros((10,10,3))
  square[:,:,0] = R
  square[:,:,1] = G
  square[:,:,2] = B
  square_hsv = rgb2hsv(square)

  h2 = np.mean(square_hsv[:,:,0])
  s2 = np.mean(square_hsv[:,:,1])
  v2 = np.mean(square_hsv[:,:,2])
  
  # print(f"Square HSV {round(h2,2)} {round(s2,2)} {round(v2,2)}")

  # Differences
  dh = h1 - h2
  ds = s1 - s2
  dv = v1 - v2
  
  # Euclidean distance in 3D HSV space
  hsv_distance = np.sqrt( (h1-h2)**2 + (s1-s2)**2 + (v1-v2)**2)


  return im, patch, hsv_distance, h1,s1,v1,h2,s2,v2,dh,ds,dv



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
    
    

    cr = colours[trial_index][0]
    cg = colours[trial_index][1]
    cb = colours[trial_index][2]
    im2, patch,d = place_square(image,cr,cg,cb)

    show(im2)
    show(patch)
    print(f"HSV distance = {d}")
    hsv_distances.append(d)
    im2 = skimage.img_as_ubyte(im2)
    imsave(folder+f"trial_{trial_number}_frame_{frame_number:04}_s_{seconds}.png", im2)

    trial_index += 1
    if trial_index > MAX_TRIALS:
      go = False
      
  return hsv_distances

def run_analysis_from_csv_all_participants(boxes):# Loop over the images, loading them and doing the analysis
# With float images so can use NaN
# Also write out to file
  hsv_distances = []
  trial_details = []

  delay = 5
  initial_delay = 5

  # Loop over the images, loading them and doing the analysis
  go = True

  box_index = 0
  subject_index = 0

  trial_index = 0
  prev_id = None
  prev_video = None

  while go:
    #show(image)
    if box_index >= len(boxes):
      break

    this_box = boxes[box_index]
    box_name = this_box['box_name']
    this_id = this_box['id']

    video = this_box['video']

    if video == 'white' or video == 'black':
      box_index += 1
      continue


    if prev_video is not None:
      if prev_video != video:
        print('New video!')
        trial_index = 0

    folder = 'data/frames/' + video + '/'

    seconds = initial_delay + (trial_index * delay)
    trial_number = trial_index + 1
    frame_number = seconds * 30 # fps = 30

   

    filename = folder + f"frame{frame_number:04}.jpg"
    # print(filename)
    # if exists(filename):
    #   image = imread(filename)
    #   image = skimage.img_as_float(image)
    # else:
    #   # File was not found here!
    #   # KEY UPDATES
    #   print('FILE NOT FOUND')
    #   trial_index += 1

    #   box_index += 1
    #   prev_id = this_id
    #   prev_video = video
    #   continue

    try:
      image = imread(filename)
      image = skimage.img_as_float(image)
    except Exception as e:
      # File was not found here!
      # KEY UPDATES
      print('FILE NOT FOUND')
      trial_index += 1

      box_index += 1
      prev_id = this_id
      prev_video = video
      continue

    if prev_id is not None:
      if prev_id != this_id:
        # We have moved on to a new subject
        subject_index += 1
        trial_index = 0

    colour = COLOURS[box_name]

    cr = colour[0]
    cg = colour[1]
    cb = colour[2]
    # RGB values need to be 0-1
    cr = cr/255
    cg = cg/255
    cb = cb/255
    im2, patch,d, h1,s1,v1,h2,s2,v2,dh,ds,dv = place_square(image,cr,cg,cb)

    # show(im2)
    # show(patch)

    # print(box_index, subject_index, video)
    # print(f"HSV distance = {d}")
    hsv_distances.append(d)
    im2 = skimage.img_as_ubyte(im2)
    # imsave(folder+f"trial_{trial_number}_frame_{frame_number:04}_s_{seconds}.png", im2)
    print(f"{box_index} - trial {trial_number}: frame {frame_number} at {seconds} seconds {filename} {video} {box_name}    {this_id}")

    # KEY UPDATES
    trial_index += 1

    box_index += 1
    prev_id = this_id
    prev_video = video

    # print('NEXT!')

    trial_details.append({'box_index': box_index,
      'trial_index':trial_index,
      'video': video,
      'box_colour': colour,
      'box_name': box_name,
      'hsv_distance': d,
      'dh': dh,
      'ds': ds,
      'dv':dv,
      'pid': this_id,
      'h1':h1,
      's2':s2,
      'v1':v1,
      'h2':h2,
      's1':s1,
      'v2':v2,

      })

 

      
  return hsv_distances, trial_details


