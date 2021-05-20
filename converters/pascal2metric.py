import xml.etree.ElementTree as ET
from optparse import OptionParser
import re
from glob import iglob
import os

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


if __name__=="__main__":
    """
        converts Pascal VOC files to new format that used by metrics evaluator
    """

    parser = OptionParser()
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir")
    (options, args) = parser.parse_args()
    gt_folder = options.dir
    bb_gt_dir = "/".join(re.split(r'/', gt_folder)[:-1]) + "/metric_est_format"

    try:
        os.makedirs(bb_gt_dir)
    except Exception as e:
        pass

    for filename in os.listdir(gt_folder):
        if filename.endswith(".xml"):
            bb_list = read_content(gt_folder+"/"+filename)
            with open(bb_gt_dir+"/"+filename[:-4]+".txt", 'w') as f:
                for line in bb_list:
                    prep_str = f"person {line[0]} {line[1]} {line[2]} {line[3]}\n"
                    f.write(prep_str)