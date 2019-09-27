import ML_utils as mlu
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_crack\\"
data_dir = main_dir+"data\\Quality_data\\"

chopsize = 224
offset = (40,40)
crack_thresh_overlap = 0.25
nocrack_thresh_overlap = 0.10
crack_thresh_area = 0.05
nocrack_thresh_area = 0.01
crack_thresh_dist = 20
crack_thresh_len = 0.8*chopsize

crop_directory = os.path.join(data_dir,"cropped_images_v2")
if not os.path.isdir(crop_directory):
    os.system("mkdir %s"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete_cracked"):
    os.system("mkdir %s\\concrete_cracked"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete_cracked\\mask"):
    os.system("mkdir %s\\concrete_cracked\\mask"%crop_directory)

list_paths = []
for impath in os.listdir(data_dir):
    if ((impath.endswith("jpg")) | (impath.endswith("JPG"))): list_paths.append(impath)
lanscape_crack = ["fissures_0053.JPG", "fissures_0054.JPG",
                  "fissures_0078.JPG", "fissures_0083.JPG", "fissures_0084.JPG"]
for image_name in list_paths:
    image_path = os.path.join(data_dir, image_name)
    mask_path = image_path[0:len(image_path) - 4] + "_mask.png"
    img = Image.open(image_path)
    ## MASK IMAGE WITH CRACK LOCATION
    crk_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2GRAY)
    # crk_mask = cv2.erode(crk_mask,np.ones((3,3)),iterations=1)
    crk_mask = Image.fromarray(np.array(crk_mask > 0))
    out_name = image_name.split(".")[0]
    if image_name in lanscape_crack:
        im_crop, box_crop = mlu.resample_image(img, chopsize=chopsize, offset=(offset[1], offset[0]))
    else:
        im_crop, box_crop = mlu.resample_image(img, chopsize=chopsize, offset=offset)
    print("%s divided into %i" % (out_name, len(im_crop)))
    # find all proper crack images
    print("Searching for cracks")
    dist = np.arange(chopsize) - (chopsize - 1) / 2
    distx, disty = np.meshgrid(dist, dist)
    dist = np.sqrt(distx ** 2 + disty ** 2)
    img_mask = np.zeros((img.size[1],img.size[0]))
    ## [1] DETECT CRACK AND SAVE THE CENTERED PATCHES
    for j in range(len(im_crop)):
        x0 = int(box_crop[j][0])
        y0 = int(box_crop[j][1])
        out = out_name + "_crop_%i_%i" % (y0,x0)
        sub_crack = np.asarray(crk_mask.crop(box_crop[j]))
        sub_over = img_mask[y0:y0+chopsize,x0:x0+chopsize]
        # IF SUFFICIENT PORTION OF THE PATCH IS COVERED BY A CRACK
        is_crack = ((np.sum(sub_crack)/(chopsize*chopsize))>crack_thresh_area)
        # IF AUTHORIZED OVERLAP WITH PREVIOUSLY SELECTED CRACK IMAGE PATCHES
        is_crack = is_crack & (np.sum(sub_over)/(chopsize*chopsize)<crack_thresh_overlap)

        if is_crack:
            # COMPUTE THE LENGTH OF THE CRACK
            len_crack = max([np.size(np.where(np.sum(sub_crack,axis=0)>0)),
                             np.size(np.where(np.sum(sub_crack,axis=1)>0))])
            #print("Crack length in pix %i"%len_crack)
            # IF THE CRACK IS CENTERED & LONG ENOUGH
            if ((np.min((dist*sub_crack)[np.where(sub_crack>0)]) < crack_thresh_dist) &
                    (len_crack>crack_thresh_len)):
                # if THERE IS a crack AND if PERMITTED OVERLAP with previously selected patches
                img_mask[y0:y0+chopsize, x0:x0+chopsize] = 1
                im_crop[j].save(os.path.join(crop_directory+"\\concrete_cracked",out+".jpg"))
                sub = Image.fromarray(sub_crack)
                sub.save(os.path.join(crop_directory+"\\concrete_cracked\\mask",out+".jpg"))
    #fig = plt.figure(0)
    #fig.clear()
    #ax = fig.add_subplot(111)
    #ax.imshow(img_mask)
    #plt.show()