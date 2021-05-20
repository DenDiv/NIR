from mtcnn import MTCNN
import cv2
from optparse import OptionParser
import re
from glob import iglob
import os
import numpy as np
from tqdm import tqdm


def usual_voc_format(im_folder):
    est_metr_dir = "/".join(re.split(r'/', im_folder)[:-1]) + "/annotations/metric_est_format_mtcnn"
    try:
        os.makedirs(est_metr_dir)
    except Exception as e:
        pass

    for filename in tqdm(os.listdir(im_folder)):
        if filename.endswith(".jpg"):
            img = cv2.cvtColor(cv2.imread(im_folder + "/" + filename), cv2.COLOR_BGR2RGB)
            res = detector.detect_faces(img)
            with open(est_metr_dir + "/" + filename[:-4] + ".txt", 'w') as f:
                for frame in res:
                    f.write(
                        f"person {frame['confidence']} {frame['box'][0]} {frame['box'][1]} {frame['box'][0] + frame['box'][2]} {frame['box'][1] + frame['box'][3]}\n")


def WIDER_voc_format(im_folder, start_image_name):
    try:
        os.makedirs("data/WIDER/pred_mtcnn")
    except Exception as e:
        pass

    start_flag = False
    if start_image_name is None:
        start_flag = True
    for event in tqdm(os.listdir(f"{im_folder}/images")):
        try:
            os.makedirs(f"data/WIDER/pred_mtcnn/{event}")
        except Exception as e:
            pass
        for filename in tqdm(os.listdir(f"{im_folder}/images/{event}")):
            if f"{im_folder}/images/{event}/{filename}" == start_image_name:
                start_flag=True
            if start_flag==False:
                continue
            if filename.endswith(".jpg"):
                img = cv2.cvtColor(cv2.imread(f"{im_folder}/images/{event}/{filename}"), cv2.COLOR_BGR2RGB)
                res = detector.detect_faces(img)
                with open(f"./eval_tools/pred_mtcnn/{event}/{filename[:-4]}" + ".txt", 'w') as f:
                    f.write(f"{filename[:-4]}\n{len(res)}\n")
                    for frame in res:
                        f.write(
                            f"{frame['box'][0]} {frame['box'][1]} {frame['box'][2]} {frame['box'][3]} {frame['confidence']}\n")


if __name__ == "__main__":
    """
        if not WIDER dataset - format: person, confidence, left, top, right, down, 
        if WIDER - format: left, top, width, height, confidence
    """
    parser = OptionParser()
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir")
    parser.add_option("-m", "--mode", action="store", type="string", dest="mode")
    parser.add_option("-s", "--start_image", action="store", type="string", dest="start")
    (options, args) = parser.parse_args()
    im_folder = options.dir
    mode = options.mode
    start_image = options.start
    detector = MTCNN()

    if mode != "WIDER":
        usual_voc_format(im_folder)
    else:
        WIDER_voc_format(im_folder, start_image)