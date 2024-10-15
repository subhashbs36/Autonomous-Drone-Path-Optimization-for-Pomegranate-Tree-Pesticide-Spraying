from dataclasses import dataclass
from sys import displayhook
import cv2
from pathlib import Path
from django.conf import settings
import os, shutil
from empatches import EMPatches
import torch 
import pandas as pd
import PIL
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import math
from itertools import permutations
from timeit import default_timer as timer  
import PIL
from PIL import Image, ImageDraw, ImageFont

PIL.Image.MAX_IMAGE_PIXELS = 933120000

from detect import detectMain

EARTH_RADIUS = 6371000

def downsizeImage1(image1_path,crop):
    # print(crop)
    if crop == False:
        img_loc = image1_path
        print(img_loc)
        img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
    else: 
        img = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    dest = 'Resisedimages'
    try:
        os.chdir(dest)
    except:
        os.mkdir(dest)
    if crop == False:
        name = image1_path.split('.')[0]+'.jpg'
    else:
        name = 'resized.jpg'
    t = cv2.imwrite(name, resized)
    print(t)
    return name, img.shape


def slicingImage1(img_loc,img_id, slice_loc):
    img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
    def conv(iter):
        i = indices.index(iter)
        sliced_index_dict['img{}'.format(i)] = iter

    emp = EMPatches()
    img_patches, indices = emp.extract_patches(img, patchsize=1024, overlap=0.4)
    sliced_index_dict={}
    for iter in indices:
        conv(iter)
    i = 0
    dest = f'{slice_loc}/{img_id}/'
    try:
        os.chdir(dest)
    except:
        os.mkdir(dest)
        os.chdir(dest)
    new_dest = os.getcwd()
    for img_patche in img_patches:
        cv2.imwrite(f'img{i}.jpg', img_patche)
        i = i + 1
    return  sliced_index_dict, new_dest

def detectTrees(weights_loc, img_id, conf, save_conf):
    source_loc = img_id
    model_result_bbox = detectMain(weights_loc,source_loc,conf,save_conf) 
    return model_result_bbox


def convertBboxToGlobal(model_result,slices_index_dict):
    bb = []
    dw = 1024
    dh = 1024
    for img, value in model_result.items():
    # xmin,ymin coordinates of sliced image
        slice_img_x = slices_index_dict[img][2]
        slice_img_y = slices_index_dict[img][0]
        # iterate through trees
        for tree, box in value.items():
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            conf = box[4]
            # denormalise
            x1 = ((x-w/2) * dw)
            x2 = ((x+w/2) * dw)
            y1 = ((y-h/2) * dh)
            y2 = ((y+h/2) * dh)
            # convert boxes
            xstart = int(x1)+slice_img_x
            ystart = int(y1)+slice_img_y
            xend = int(x2)+slice_img_x
            yend = int(y2)+slice_img_y
            bb.append([xstart,xend,ystart,yend,conf])
            
    P = torch.tensor(bb)
    iou_thresh = 0.01
    final_boxes = removeRedundantBboxes(P,iou_thresh)
    return final_boxes

def removeRedundantBboxes(P : torch.tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # we extract coordinates for every prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 2]
    x2 = P[:, 1]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for filtered prediction boxes
    keep = []
    while len(order) > 0:
        # extract the index of the prediction with highest score we call this prediction S
        idx = order[-1]
        # push S in filtered predictions list
        d = {'bbox':{'xmin':int(P[idx][0]),'xmax':int(P[idx][1]),'ymin':int(P[idx][2]),'ymax':int(P[idx][3])},'height':None,'diameterr':None,'yield':None} 
        keep.append(d)
        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break
        # select coordinates of BBoxes according to the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        # take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        # find the intersection area
        inter = w*h
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
        # find the union of every prediction T in P with the prediction S. Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        # find the IoU of every prediction in P with S
        IoU = inter / union
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep

def plotFinalBoxes(final_boxes,image1_path, dest_folder, img_name):
    img_loc = image1_path
    img = cv2.imread(img_loc, cv2.IMREAD_UNCHANGED)
    for box in final_boxes:
        cv2.rectangle(img, (box['bbox']['xmin'], box['bbox']['ymin']), (box['bbox']['xmax'], box['bbox']['ymax']), (0,0,255), 6)
    dest = dest_folder
    try:
        os.chdir(dest)
    except:
        os.mkdir(dest)
    name = f'{dest}/{img_name}'
    cv2.imwrite(name,img)
    return name

def getGSD(img):
    with rio.open(img) as src:
    # Get the GSD values
        metadata = src.meta

    # Extract the GSD value
        gsd = metadata['transform'][0]

    # Print the results
    return gsd

def extractHeight(final_boxes,img1_path, img2_path):
    imgdata = rio.open(img2_path)
    img = imgdata.read(1)
    i = 0
    keep = []
    for items in final_boxes:
        value = items["bbox"]
        xmin = int(value['xmin']/0.6)
        xmax = int(value['xmax']/0.6)
        ymin = int(value['ymin']/0.6)
        ymax = int(value['ymax']/0.6)
        hmax=0
        hmin=10000
        imgGSD = getGSD(img1_path)
        diameter = ((abs(xmax-xmin)+abs(ymax-ymin))/2)*imgGSD
        items['diameterr'] = diameter

        for x in range(xmin,xmax):
            for y in range(ymin,ymax):
                if img[y][x] < hmin:
                    hmin = img[y][x] 
                elif img[y][x] > hmax:
                    hmax = img[y][x] 

        tree_height = hmax - hmin
        items["height"] = tree_height
        i+=1
        d = {'bbox':{'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax},'height':tree_height,'diameter':diameter, 'latitude':None, 'longitude':None} 
        keep.append(d)

    return keep

def geoCordinates(img_path, x, y):
    # Open the TIFF file
    with rio.open(img_path) as src:
        # Get the georeferencing information
        transform = src.transform
        # Convert the x, y coordinates to latitude and longitude
        lon, lat = transform * (x, y)
        return lat, lon

def geoCordinates_of_OriginalImage(image1_path, final_boxes):
    keep = []
    for items in final_boxes:
        value = items["bbox"]
        xmin = value['xmin']
        xmax = value['xmax']
        ymin = value['ymin']
        ymax = value['ymax']

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        keep.append((x_center,y_center))
        lat, long = geoCordinates(image1_path, x_center, y_center)
        items['latitude'] = lat
        items['longitude'] = long

    return final_boxes, keep


#path Visualizer
def  visualizer(path, img):
    # Load the image
    image = Image.open(img)

    # Define colors
    point_color = (255, 0, 0)
    line_color = (0, 0, 255)
    text_color = (0, 255, 0)

    # Define the font for labeling the points
    font = ImageFont.truetype("arial.ttf", 25)

    # Create a copy of the image to draw on
    draw_image = image.copy()

    # Create a draw object
    draw = ImageDraw.Draw(draw_image)

    # Draw the path
    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=15)

    # Draw the points and labels
    for i, (x, y) in enumerate(path):
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=point_color)
        draw.text((x+10, y-10), str(i), fill=text_color, font=font)

    # Save the image
    draw_image.save("path_with_labels.png")

    return draw_image




#Path Finder Program modules below

def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def calculate_path_length(points, path):
    return sum(distance(points[path[i]], points[path[i+1]]) for i in range(len(path)-1))

def two_opt(points, path):
    n = len(points)
    while True:
        improvement = 0
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j-i == 1:
                    continue
                new_path = path[:]
                new_path[i:j] = path[j-1:i-1:-1]
                new_length = calculate_path_length(points, new_path)
                if new_length < calculate_path_length(points, path):
                    path = new_path
                    improvement = 1
        if improvement == 0:
            break
    return path

def PathFinder_tsp_2opt(path, start, end):
    start_position = start
    end_position = end

    # Add the start and end positions to the list of points
    path.insert(0, start_position)
    path.append(end_position)

    # Find the optimal path and calculate its length
    initial_path = [i for i in range(len(path))]
    optimal_path = two_opt(path, initial_path)
    optimal_path_length = calculate_path_length(path, optimal_path)

    optimal_path_noStrip = optimal_path
    # Remove the start and end positions from the optimal path
    optimal_path = optimal_path[1:-1]

    # Print the results
    print("Optimal Path:", optimal_path)
    print("Optimal Path Length:", optimal_path_length)

    return optimal_path, optimal_path_length, optimal_path_noStrip







#########--------------######################

weight_loc = r'main_best.pt'
resized_img = r'resized.jpg'
image1_path = r'OriginalImages\yp_upload3.tif'
image2_path = r'OriginalImages\yp_upload4.tif'
slice_loc = r'sliced_images'
FinalImage = r'FinalImage'
path_save =r'demo.txt'
start = (0,0)
end=(0,0)

# image1_path = crop_img_path
img_id = 1
crop = True
yoloThreshold = 0.7
save_images = True          #Yolo save Images

 #Emptying the slice folder for next image-- not for production
folder = slice_loc
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))




#downsampling
resized_name, dim = downsizeImage1(image1_path,crop)
#slicing of images 
image_Slice_dict, sliceFolder_loc = slicingImage1(resized_img,img_id, slice_loc)
# print(image_Slice_dict)

#yolo
current_dir = os.getcwd()
os.chdir(current_dir)
model_results = detectTrees(weight_loc, sliceFolder_loc, yoloThreshold, save_images)
print("----------------\n")
# print(model_results)

resized_box = convertBboxToGlobal(model_results, image_Slice_dict)
# print(resized_box)

#Plot bbox on Resisedimages 
procesedImage =  plotFinalBoxes(resized_box, resized_img.replace('tif', 'jpg'), FinalImage, resized_name)
# print(procesedImage)
print("\nDone with Resized Image - Data Extraction Process")
print("------------------------\n")

#Getting the Original Image bbox
final_box_without_geo = extractHeight(resized_box, image1_path, image2_path)

final_box_with_geo, center_coord = geoCordinates_of_OriginalImage(image1_path, final_box_without_geo)
print("Done with Original Image- Geo Location data extraction")
print("------------------------\n")
print(final_box_with_geo)
print("\nPath Calculation Started... \nGoing to Take a long time so sit back or come back after a while... \n")
time_taken_path_calculation = timer()


optimal_path, optimal_path_length, optimal_path_noStip = PathFinder_tsp_2opt(center_coord,start, end)


x,y = start
latStart, lonStart = geoCordinates(image1_path, x, y)
j,k = end
latEnd, lonEnd = geoCordinates(image1_path, j, k)

new_path = []
for i in optimal_path:
    a = final_box_with_geo[i-1]
    new_path.append(a)

beggining = {'height': 2.5394287, 'latitude': latStart, 'longitude': lonStart}
Finish = {'height': 2.5394287, 'latitude': latEnd, 'longitude': lonEnd}

new_path.insert(0, beggining)
new_path.append(Finish)

print(new_path)

# open the file for writing
with open(path_save, 'w') as f:
    # iterate over the dat
    f.write(str(new_path) + '\n')

print("\nPath Proces Ended..")
print("Time Taken to calculate: ", (timer()-time_taken_path_calculation)/60)

# Visualization
print("\n----Entering Image Visualization Process...\nThis can take 2-4min to complete....\n ")

img_optimal_path = []
for i in optimal_path:
    a = center_coord[i]
    img_optimal_path.append(a)

img_optimal_path.insert(0, start)
img_optimal_path.append(end)

img = visualizer(img_optimal_path, image1_path)
print("Final Image Has Been Saved")
img.show()
