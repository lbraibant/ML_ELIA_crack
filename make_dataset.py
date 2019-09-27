import ML_utils as mlu
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion


main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_pylons\\"
data_dir = main_dir+"data\\cracked_annotated\\"
# Add mask to the json file that contains annotations
# metadata, regions = read_json_annotated_images(main_dir+"via_project_fissures.json")
# out_metadata = main_dir+"new_via_project_fissures.json"
# metadata = add_masks_to_metadata(main_dir+"via_project_fissures.json",
#                                  main_dir+"data/TIF/", json_out=out_metadata,
#                                  region_name="crack")
metadata, regions = mlu.read_json_annotated_images(data_dir+"concrete_pylon_crack_annot_via.json")
print("regions")
print(len(metadata),metadata[0].keys())
chopsize = 224
portraits = ["0053","0055","0057","0058","0059","0060","0061","0062","0063","0064",
             "0065","0066","0067","0068","0069","0070","0071","0072","0073","0074",
             "0075","0077","0078","0079","0080","0081","0082","0083","0084","0085",
             "0086","0087","0088","0089","0090","0091","0092","0095","0099","0100",
             "0101","0102","0104","0105","0106","0107","0109","0110","0111","0112",
             "0113","0114","0115","0116","0117","0118","0119","0120","0121","0122",
             "0123","0124","0125","0126","0127","0128","0129","0130","0131","0132",
             "0133","0134","0135","0136","0138","0139"]
crop_directory = os.path.join(data_dir,"cropped_images")
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
#for image_num in range(len(metadata)):
for image_num in [0]:
    image_path = metadata[image_num]["filename"].strip()
    mask_path = image_path[0:len(image_path)-4]+"_mask.tif"
    image_path = image_path[0:len(image_path)-4]+"_full.JPG"
    img = Image.open(image_path)
    fig_0 = plt.figure(0)
    fig_0.clear()
    ax = fig_0.add_subplot(111)
    mlu.show_annotated_image(ax,metadata[image_num],['crack','concrete pylon'],['r','y'])
    ## Mask with cracks
    crk_mask = Image.fromarray(mlu.convert_contour_into_mask(metadata[image_num], 'crack', 'polygon'))
    #crk_mask = cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_RGB2GRAY)
    #crk_mask = cv2.erode(crk_mask,np.ones((3,3)),iterations=1)
    #crk_mask = Image.fromarray(np.array(crk_mask>0))
    ## Mask with concrete pylon
    pyl_mask = mlu.convert_contour_into_mask(metadata[image_num], 'concrete pylon', 'polygon')
    if np.sum(pyl_mask)==0:
        pyl_mask = mlu.convert_contour_into_mask(metadata[image_num], 'concrete pylon', 'rect')
    pyl_mask = Image.fromarray(pyl_mask)
    out_name = image_path.split("\\")[-1]
    out_name = out_name.split(".")[0]
    if (out_name.split("_")[1] in portraits) & (img.size[0]>img.size[1]):
        crk_mask = crk_mask.resize((img.size[1],img.size[0]),resample=Image.NEAREST)
        pyl_mask = pyl_mask.resize((img.size[1],img.size[0]),resample=Image.NEAREST)
        print(out_name.split("_")[1])
        if out_name.split("_")[1] in ["0084","0119","0105"]:
            crk_mask = Image.fromarray(np.transpose(np.asarray(crk_mask))[::-1,:])
            pyl_mask = Image.fromarray(np.transpose(np.asarray(pyl_mask))[::-1,:])
        else:
            crk_mask = Image.fromarray(np.transpose(np.asarray(crk_mask))[:,::-1])
            pyl_mask = Image.fromarray(np.transpose(np.asarray(pyl_mask))[:,::-1])
    else:
        crk_mask = crk_mask.resize(img.size,resample=Image.NEAREST)
        pyl_mask = pyl_mask.resize(img.size,resample=Image.NEAREST)
        binary_erosion
    im_crop,box_crop = mlu.resample_image(img, chopsize=chopsize, offset=int(chopsize/10))
    print("%s divided into %i"%(out_name,len(im_crop)))
    fig_1 = plt.figure(1)
    fig_1.clear()
    ax = fig_1.add_subplot(121)
    ax.imshow(img)
    axm = fig_1.add_subplot(122)
    axm.imshow(5*np.array(crk_mask)+np.array(pyl_mask))
    #plt.show()
    fig_0.savefig("all_annot_%s.png"%out_name.split("_")[1])
    fig_1.savefig("crack_full_mask_%s.png"%out_name.split("_")[1])
    # find all proper crack images
    print("Looking for cracks")
    dist = np.arange(chopsize)-(chopsize-1)/2
    distx,disty = np.meshgrid(dist,dist)
    dist = np.sqrt(distx**2+disty**2)
    img_mask = np.zeros(img.size)
    #thresoverlap = 1
    for j in range(len(im_crop)):
        out = out_name + "_crop_%i_%i" % (box_crop[j][1], box_crop[j][0])
        x0 = int(box_crop[j][0])
        y0 = int(box_crop[j][1])
        sub = np.asarray(crk_mask.crop(box_crop[j]))
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





