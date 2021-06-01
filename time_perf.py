import os
from mtcnn import MTCNN
import time
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

resolution_dirs = "data/different_resolution_img"
n_times = 20

if __name__ == "__main__":
    detector = MTCNN()
    with open("results/time_estimation/time_eval.txt", 'w') as f:
        f.write("resolution alg min median max\n")
    for res_dir in os.listdir(resolution_dirs):
        res_list = []
        for img_name in tqdm(os.listdir("/".join([resolution_dirs, res_dir]))):
            for _ in range(n_times):
                img = cv2.cvtColor(cv2.imread("/".join([resolution_dirs, res_dir, img_name])), cv2.COLOR_BGR2RGB)
                start_time = time.time()
                detector.detect_faces(img)
                end_time = time.time()
                res_list.append(end_time-start_time)
        plt.hist(res_list, bins=25, label=["sec/img"])
        plt.title(f"{res_dir} mtcnn")
        plt.legend()
        plt.savefig(f"results/{res_dir}_mtcnn.png")
        plt.close()
        #plt.show()
        with open("results/time_estimation/time_eval.txt", 'a') as f:
            f.write(f"{res_dir} mtcnn {np.min(res_list):.3f} {np.median(res_list):.3f} {np.max(res_list):.3f}\n")
