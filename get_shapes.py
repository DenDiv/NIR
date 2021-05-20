import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from optparse import OptionParser
from matplotlib import pyplot as plt

if __name__=="__main__":
    parser = OptionParser()
    parser.add_option("-d", "--dir", action="store", type="string", dest="dir")
    (options, args) = parser.parse_args()
    im_folder = options.dir

    shape_matr = []
    for event in os.listdir(f"{im_folder}/images"):
        for filename in os.listdir(f"{im_folder}/images/{event}"):
            if filename.endswith(".jpg"):
                img = Image.open(f"{im_folder}/images/{event}/{filename}")
                shape_matr.append(list(img.size))
    shape_matr = np.array(shape_matr)
    plt.hist(shape_matr[:, 1], bins=50)
    plt.show()