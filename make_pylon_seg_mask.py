import ML_utils as mlu
import json
import matplotlib.pyplot as plt
import os
from PIL import Image

main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_crack\\"
data_dir = main_dir+"data\\Pylone_Segmentation\\"

log_directory = os.path.join(data_dir,"log")
if not os.path.isdir(log_directory):
    os.system("mkdir %s"%log_directory)

json_path = os.path.join(data_dir,"all_pylone_segmentation_annotations_via.json")

metadata,dummy = mlu.read_json_annotated_images(json_path)

for num in range(len(metadata)):
    outname = metadata[num]["filename"]
    outname = outname.split('.')[0]
    fig = plt.figure(0)
    fig.clear()
    axim = fig.add_subplot(121)
    axim.imshow(Image.open(os.path.join(data_dir,metadata[num]["filename"])))
    axma = fig.add_subplot(122)
    mask,axma = mlu.convert_contour_into_mask(metadata[num], "concrete pylon", "polygon",
                                              directory=data_dir, ax=axma)
    fig.savefig(os.path.join(log_directory,outname+"_check_mask.png"))
    mask = Image.fromarray(mask)
    mask.save(os.path.join(data_dir,outname+"_mask.JPG"))