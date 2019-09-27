import ML_utils as mlu
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_crack\\"
data_dir = main_dir+"data\\Quality_data\\"

chopsize = 224
offset = (chopsize,40)
crack_thresh_overlap = 0.25
nocrack_thresh_overlap = 0.10
crack_thresh_area = 0.1
nocrack_thresh_area = 0.01
crack_thresh_dist = 20

crop_directory = os.path.join(data_dir,"cropped_images_v2")
if not os.path.isdir(crop_directory):
    os.system("mkdir %s"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete"):
    os.system("mkdir %s\\concrete"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete\\uncracked"):
    os.system("mkdir %s\\concrete\\uncracked"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete\\cracked"):
    os.system("mkdir %s\\concrete\\cracked"%crop_directory)
if not os.path.isdir(crop_directory+"\\concrete\\cracked\\masks"):
    os.system("mkdir %s\\concrete\\cracked\\masks"%crop_directory)
if not os.path.isdir(crop_directory+"\\background"):
    os.system("mkdir %s\\background"%crop_directory)
if not os.path.isdir(crop_directory+"\\pylon_edge"):
    os.system("mkdir %s\\pylon_edge"%crop_directory)

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
    fig_1 = plt.figure(1)
    fig_1.clear()
    ax = fig_1.add_subplot(121)
    ax.imshow(img)
    axm = fig_1.add_subplot(122)
    axm.imshow(np.array(crk_mask))
    fig_1.savefig(os.path.join("%s\\log" % crop_directory, "%s.png" % out_name))
    # find all proper crack images
    print("Searching for cracks")
    dist = np.arange(chopsize) - (chopsize - 1) / 2
    distx, disty = np.meshgrid(dist, dist)
    dist = np.sqrt(distx ** 2 + disty ** 2)
    img_mask = np.zeros(img.size)
    ## [1] DETECT CRACK AND SAVE THE CENTERED PATCHES
    for j in range(len(im_crop)):
        x0 = int(box_crop[j][0])
        y0 = int(box_crop[j][1])
        out = out_name + "_crop_%i_%i" % (y0,x0)
        sub_crack = np.asarray(crk_mask.crop(box_crop[j]))
        sub_over = img_mask[y0:y0+chopsize,x0:x0+chopsize]
        # IF SUFFICIENT PORTION OF THE PATCH IS COVERED BY A CRACK
        is_crack = (np.sum(sub_crack)/(chopsize*chopsize)>crack_thresh_area)
        # IF AUTHORIZED OVERLAP WITH PREVIOUSLY SELECTED CRACK IMAGE PATCHES
        is_crack = is_crack & (np.sum(sub_over)/(chopsize*chopsize)<crack_thresh_overlap)
        # IF THE CRACK IS CENTERED
        is_crack = is_crack & (np.min((dist*sub_crack)[np.where(sub_crack>0)]) < crack_thresh_dist)
        # if THERE IS a crack AND if PERMITTED OVERLAP with previously selected patches
        if is_crack:
            img_mask[y0:y0 + chopsize, x0:x0 + chopsize] = 1
            im_crop[j].save(os.path.join(crop_directory+"\\cracked_concrete",out+".jpg"))
            sub = Image.fromarray(sub_crack)
            sub.save(os.path.join(crop_directory+"\\cracked_concrete\\mask",out+".jpg"))
    plt.figure(0)
    plt.imshow(img_mask)
    plt.show()

    ## [2] SAMPLE IMAGE PATCHES OF UNCRACKED CONCRETE AND BACKGROUND
    for j in range(len(im_crop)):
        x0 = int(box_crop[j][0])
        y0 = int(box_crop[j][1])
        out = out_name + "_crop_%i_%i" % (y0,x0)
        sub_crack = np.asarray(crk_mask.crop(box_crop[j]))
        sub_over = img_mask[y0:y0+chopsize,x0:x0+chopsize]
        # IF SUFFICIENT PORTION OF THE PATCH IS COVERED BY A CRACK
        is_crack = (np.sum(sub_crack)/(chopsize*chopsize)>crack_thresh_area)
        # IF AUTHORIZED OVERLAP WITH PREVIOUSLY SELECTED CRACK IMAGE PATCHES
        is_crack = is_crack & (np.sum(sub_over)/(chopsize*chopsize)<crack_thresh_overlap)
        # IF THE CRACK IS CENTERED
        is_crack = is_crack & (np.min((dist*sub_crack)[np.where(sub_crack>0)]) < crack_thresh_dist)
        # if THERE IS a crack AND if PERMITTED OVERLAP with previously selected patches
        if is_crack:
            img_mask[y0:y0 + chopsize, x0:x0 + chopsize] = 1
            im_crop[j].save(os.path.join(crop_directory+"\\cracked_concrete",out+".jpg"))
            sub = Image.fromarray(sub_crack)
            sub.save(os.path.join(crop_directory+"\\cracked_concrete\\mask",out+".jpg"))
    plt.figure(0)
    plt.imshow(img_mask)
    plt.show()

        mindist = chopsize
        if (np.sum(sub)/(chopsize*chopsize)>0.05): mindist = np.min((dist*sub)[np.where(sub>0)])
        foverlap = np.sum(img_mask[y0:y0+chopsize,x0:x0+chopsize])/(chopsize*chopsize)
        #if ((foverlap<thresoverlap) & (mindist<chopsize/4)):
        if (mindist < chopsize/4):
            img_mask[y0:y0 + chopsize, x0:x0 + chopsize] = 1
            im_crop[j].save(os.path.join(crop_directory+"\\concrete\\cracked",out+".jpg"))
            sub = Image.fromarray(sub)
            sub.save(os.path.join(crop_directory+"\\concrete\\cracked\\masks",out+".jpg"))
    # for images of uncracked concrete and background, reduce overlap threshold to 20%
    print("Saving non overlaping pictures of backgorund and uncracked concrete")
    thresoverlap = 0.1
    for j in range(len(im_crop)):
        out = out_name+"_crop_%i_%i"%(box_crop[j][1],box_crop[j][0])
        x0 = box_crop[j][0]
        y0 = box_crop[j][1]
        sub = np.asarray(crk_mask.crop(box_crop[j]))
        foverlap = np.sum(img_mask[y0:y0+chopsize,x0:x0+chopsize])/(chopsize*chopsize)
        if (foverlap<thresoverlap):
            img_mask[y0:y0 + chopsize, x0:x0 + chopsize] = 1
            tot = np.sum(np.asarray(pyl_mask.crop(box_crop[j])))
            if tot/(chopsize*chopsize)>0.9:
                sub = crk_mask.crop(box_crop[j])
                tot = np.sum(np.asarray(sub))
                if tot/(chopsize*chopsize)<0.05:
                    im_crop[j].save(os.path.join(crop_directory+"\\concrete\\uncracked",out+".jpg"))
            else:
                if tot/(chopsize*chopsize)<0.2:
                    im_crop[j].save(os.path.join(crop_directory + "\\background", out + ".jpg"))
                else:
                    im_crop[j].save(os.path.join(crop_directory + "\\pylon_edge", out + ".jpg"))





