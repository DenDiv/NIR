from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from optparse import OptionParser
from tqdm import tqdm
from pascal_voc_writer import Writer
import cv2


def mafa2pascal(mat_file, save_path, del_invalid):
    try:
        os.makedirs(f"../{save_path}/annotations")
    except Exception as e:
        pass
    annotations = loadmat(f"../{mat_file}")
    key_name = list(annotations.keys())[-1]
    # masked, unmasked, invalid
    face_counter = {1: 0, 2: 0, 3: 0}
    for image in tqdm(annotations[key_name][0]):
        image_name = image[0][0]
        im_path = "../"+"/".join([save_path, "images", image_name])
        im = cv2.imread(im_path)
        h, w, _ = im.shape
        writer = Writer(im_path, h, w)
        skip_flag = False
        for face_num in range(image[1].shape[0]):
            face = image[1][face_num]
            if del_invalid and face[4] == 3:
                skip_flag = True
                break
            x_face, y_face, w_face, h_face = face[0], face[1], face[2], face[3]
            writer.addObject('person', int(x_face), int(y_face), int(x_face) + int(w_face), int(y_face) + int(h_face))
            face_counter[face[4]] += 1
        if not skip_flag:
            writer.save("../"+"/".join([save_path, "annotations", image_name[:-3]+"xml"]))
    print(f"masked_faces: {face_counter[1]}\nunmasked_faces: {face_counter[2]}\ninvalid_faces: {face_counter[3]}\ntotal_faces: {face_counter[1]+face_counter[2]+face_counter[3]}")


if __name__ == "__main__":
    """
    converter for test mafa data to pascal VOC
    """
    parser = OptionParser()
    parser.add_option("--mat_file", action="store", type="string", dest="mat_file")
    parser.add_option("--save_path", action="store", type="string", dest="save_path")
    parser.add_option("--del_invalid", action="store", type="string", default=False, dest="del_invalid")
    (options, args) = parser.parse_args()
    mat_file = options.mat_file
    save_path = options.save_path
    if options.del_invalid == "False":
        del_invalid = False
    else:
        del_invalid = True
    mafa2pascal(mat_file, save_path, del_invalid)