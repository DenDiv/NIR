from eval_lib.utils import *
from eval_lib.BoundingBox import BoundingBox
from eval_lib.BoundingBoxes import BoundingBoxes
from eval_lib.Evaluator import Evaluator
import os
from optparse import OptionParser


def getBoundingBoxes(gt_dir, pred_dir):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    # read ground truth
    gt_im_names = []
    for filename in os.listdir(gt_dir):
        nameOfImage = filename.replace(".txt","")
        gt_im_names.append(nameOfImage)
        fh1 = open(f"{gt_dir}/{filename}", "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0] #class
            x1 = float(splitLine[1])
            y1 = float(splitLine[2])
            x2 = float(splitLine[3])
            y2 = float(splitLine[4])
            bb = BoundingBox(imageName=nameOfImage, classId=idClass, x=x1, y=y1, w=x2, h=y2, typeCoordinates=CoordinatesType.Absolute,
                             bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # read detections
    for filename in os.listdir(pred_dir):
        nameOfImage = filename.replace(".txt", "")
        if nameOfImage not in gt_im_names:
            continue
        fh1 = open(f"{pred_dir}/{filename}", "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])
            x1 = float(splitLine[2])  # confidence
            y1 = float(splitLine[3])
            x2 = float(splitLine[4])
            y2 = float(splitLine[5])
            bb = BoundingBox(imageName=nameOfImage, classId=idClass, x=x1, y=y1, w=x2, h=y2,
                             typeCoordinates=CoordinatesType.Absolute, classConfidence=confidence,
                             bbType=BBType.Detected, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


if __name__ == "__main__":
    """
        used library for computing metrics from this project: https://github.com/rafaelpadilla/Object-Detection-Metrics
    """
    parser = OptionParser()
    parser.add_option("--gt", action="store", type="string", dest="gt")
    parser.add_option("--predicted", action="store", type="string", dest="predicted")
    parser.add_option("--name", action="store", type="string", dest="name")
    (options, args) = parser.parse_args()
    gt_file = options.gt
    predicted = options.predicted
    method_name = options.name
    bbxs = getBoundingBoxes(gt_file, predicted)
    evaluator = Evaluator()
    evaluator.PlotPrecisionRecallCurve(bbxs, IOUThreshold=0.5, showAP=True, showInterpolatedPrecision=False, eval_meth_name=method_name,
                                       savePath=f"results/AP_estimation/real_masked_{method_name}_PR_curve_IOU_0_5.png")