#!/usr/bin/env python3
import cv2
import os
import numpy as np
from glob import iglob # python 3.5 or newer
from shutil import copyfile
from optparse import OptionParser
import xml.etree.cElementTree as ET

# The script
curr_path = os.getcwd()

# settings
cnt = 0


def newXMLPASCALfile(imageheight, imagewidth, path, basename):
    # print(filename)
    annotation = ET.Element("annotation", verified="yes")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = basename
    ET.SubElement(annotation, "path").text = path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "test"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(imagewidth)
    ET.SubElement(size, "height").text = str(imageheight)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    tree = ET.ElementTree(annotation)
    # tree.write("filename.xml")
    return tree


def appendXMLPASCAL(curr_et_object, x1, y1, w, h, filename):
    et_object = ET.SubElement(curr_et_object.getroot(), "object")
    ET.SubElement(et_object, "name").text = "face"
    ET.SubElement(et_object, "pose").text = "Unspecified"
    ET.SubElement(et_object, "truncated").text = "0"
    ET.SubElement(et_object, "difficult").text = "0"
    bndbox = ET.SubElement(et_object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x1)
    ET.SubElement(bndbox, "ymin").text = str(y1)
    ET.SubElement(bndbox, "xmax").text = str(x1+w)
    ET.SubElement(bndbox, "ymax").text = str(y1+h)
    filename = filename.strip().replace(".jpg",".xml")
    curr_et_object.write(filename)
    return curr_et_object


def readAndWrite(bbx_gttxtPath, min_pixels):
    cnt = 0
    with open(bbx_gttxtPath, 'r') as f:
        curr_filename = ""
        curr_path = ""

        curr_et_object = ET.ElementTree()
        img = np.zeros((80, 80))
        for line in f:
            inp = line.split(' ')

            if len(inp)==1:
                img_path = inp[0]
                img_path = img_path[:-1]
                curr_img = img_path
                if curr_img.isdigit():
                    continue

                img = cv2.imread(im_path + '/' + curr_img, 2) # POSIX only
                curr_filename = curr_img.split("/")[1].strip()
                curr_path = os.path.join(im_path, os.path.dirname(curr_img))
                curr_et_object = newXMLPASCALfile(img.shape[0],img.shape[1],curr_path, curr_filename)


            else:
                inp = [int(i) for i in inp[:-1]]
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = inp
                n = max(w, h)
                #if invalid == 1 or blur > 0 or n < min_pixels:
                #    continue
                cnt += 1

                fileNow = os.path.join(curr_path, curr_filename)
                print("{} {} {} {}".format(x1, y1, w, h) + " " + fileNow)

                curr_et_object = appendXMLPASCAL(curr_et_object, x1, y1, w, h, fileNow )


if __name__ == "__main__":
    """
        converts source WIDER data format to Pascal VOC
        this file was copied and changed from https://github.com/qdraw/tensorflow-face-object-detector-tutorial
    """
    parser = OptionParser()
    parser.add_option("-i", "--images", action="store", type="string", dest="images")
    parser.add_option("-d", "--description", action="store", type="string", dest="description")
    parser.add_option("-m", "--mode", action="store", type="string", dest="mode")
    parser.add_option("-f", "--f", action="store", type="int", dest="filter")
    (options, args) = parser.parse_args()
    im_path = options.images
    min_pix_size = options.filter
    bbx_gttxtPath = options.description
    mode = options.mode
    readAndWrite(bbx_gttxtPath, min_pix_size)


    # To folders:
    to_xml_folder = os.path.join(curr_path, "data", f"tf_wider_{mode}", "annotations", "xmls")
    to_image_folder = os.path.join(curr_path, "data", f"tf_wider_{mode}", "images")

    # make dir => wider_data in folder
    try:
        os.makedirs(to_xml_folder)
        os.makedirs(to_image_folder)
    except Exception as e:
        pass

    rootdir_glob = im_path + '/**/*' # Note the added asterisks # This will return absolute paths
    file_list = [f for f in iglob(rootdir_glob, recursive=True) if os.path.isfile(f)]

    train_annotations_index = os.path.join(curr_path, "data", f"tf_wider_{mode}", "annotations", f"{mode}.txt" )

    with open(train_annotations_index, "a") as indexFile:
        for f in file_list:
            if ".xml" in f:
                print(f)
                copyfile(f, os.path.join(to_xml_folder, os.path.basename(f) ))
                img = f.replace(".xml",".jpg")
                copyfile(img, os.path.join(to_image_folder, os.path.basename(img) ))
                indexFile.write(os.path.basename(f.replace(".xml","")) + "\n")
