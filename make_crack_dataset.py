import ML_utils as mlu
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_pylons\\"
data_dir = main_dir+"data\\Quality_Data\\"
chopsize = 224
offset=(40,100)
np.random.seed(40)
crop_directory = os.path.join(data_dir,"cropped_images")
if not os.path.isdir(crop_directory):
    os.system("mkdir %s"%crop_directory)
if not os.path.isdir(crop_directory+"\\log"):
    os.system("mkdir %s\\log"%crop_directory)
if not os.path.isdir(crop_directory+"\\cracked_concrete"):
    os.system("mkdir %s\\cracked_concrete"%crop_directory)
if not os.path.isdir(crop_directory+"\\cracked_concrete\\mask"):
    os.system("mkdir %s\\cracked_concrete\\mask"%crop_directory)
if not os.path.isdir(crop_directory+"\\cracked_concrete_rotate"):
    os.system("mkdir %s\\cracked_concrete_rotate"%crop_directory)
if not os.path.isdir(crop_directory+"\\cracked_concrete_rotate\\mask"):
    os.system("mkdir %s\\cracked_concrete_rotate\\mask"%crop_directory)
list_paths = []
for impath in os.listdir(data_dir):
    if ((impath.endswith("jpg")) | (impath.endswith("JPG"))): list_paths.append(impath)
lanscape_crack = ["fissures_0053.JPG","fissures_0054.JPG",
                  "fissures_0078.JPG","fissures_0083.JPG","fissures_0084.JPG"]
for image_name in list_paths:
    image_path = os.path.join(data_dir,image_name)
    mask_path = image_path[0:len(image_path)-4]+"_mask.png"
    img = Image.open(image_path)
    ## Mask with cracks
    crk_mask = cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_RGB2GRAY)
    #crk_mask = cv2.erode(crk_mask,np.ones((3,3)),iterations=1)
    crk_mask = Image.fromarray(np.array(crk_mask>0))
    out_name = image_name.split(".")[0]
    if image_name in lanscape_crack:
        im_crop,box_crop = mlu.resample_image(img, chopsize=chopsize, offset=(offset[1],offset[0]))
    else:
        im_crop,box_crop = mlu.resample_image(img, chopsize=chopsize, offset=offset)
    print("%s divided into %i"%(out_name,len(im_crop)))
    fig_1 = plt.figure(1)
    fig_1.clear()
    ax = fig_1.add_subplot(121)
    ax.imshow(img)
    axm = fig_1.add_subplot(122)
    axm.imshow(np.array(crk_mask))
    fig_1.savefig(os.path.join("%s\\log"%crop_directory,"%s.png"%out_name))
    # find all proper crack images
    print("Searching for cracks")
    dist = np.arange(chopsize)-(chopsize-1)/2
    distx,disty = np.meshgrid(dist,dist)
    dist = np.sqrt(distx**2+disty**2)
    img_mask = np.zeros(img.size)
    for j in range(len(im_crop)):
        out = out_name + "_crop_%i_%i" % (box_crop[j][1], box_crop[j][0])
        x0 = int(box_crop[j][0])
        y0 = int(box_crop[j][1])
        sub = np.asarray(crk_mask.crop(box_crop[j]))
        if (np.sum(sub)/(chopsize*chopsize)>0.05):
            if (np.min((dist*sub)[np.where(sub>0)]) < chopsize/6):
                img_mask[y0:y0 + chopsize, x0:x0 + chopsize] = 1
                im_crop[j].save(os.path.join(crop_directory+"\\cracked_concrete",out+".jpg"))
                sub = Image.fromarray(sub)
                sub.save(os.path.join(crop_directory+"\\cracked_concrete\\mask",out+".jpg"))
                # rotate centered image
                add = min([min([x0,chopsize/2]),min([chopsize/2,img.size[0]-(x0+chopsize)]),
                           min([y0,chopsize/2]),min([chopsize/2,img.size[1]-(y0+chopsize)])])
                if add>(chopsize/4):
                    box = (x0-add,y0-add,x0+chopsize+add,y0+chopsize+add)
                    angle = int(np.random.uniform(0,360))
                    rot = (img.crop(box)).rotate(angle)
                    rot = rot.crop((add,add,add+chopsize,add+chopsize))
                    rot.save(os.path.join(crop_directory+"\\cracked_concrete_rotate",out+"_rot%i.jpg"%angle))
                    rot = (crk_mask.crop(box)).rotate(angle)
                    rot = rot.crop((add,add,add+chopsize,add+chopsize))
                    rot.save(os.path.join(crop_directory+"\\cracked_concrete_rotate\\mask",out+"_rot%i.jpg"%angle))

