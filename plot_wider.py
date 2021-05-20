from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
import numpy as np
import h5py
import re
plot_dir = "WIDER_eval_tools/plot/baselines/Val/setting_int"


def compute_AP(rec, prec):
    """
        function was taken from https://github.com/wondervictor/WiderFace-Evaluation/blob/master/evaluation.py
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == "__main__":
    fig_1, axs_1 = plt.subplots(1)
    fig_2, axs_2 = plt.subplots(1)
    fig_3, axs_3 = plt.subplots(1)
    for fig in fig_1, fig_2, fig_3:
        fig.set_size_inches(18.5, 10.5)
    legend_1, legend_2, legend_3 = [], [], []

    for meth in os.listdir(plot_dir):
        for file in os.listdir("/".join([plot_dir, meth])):
            if file == '.DS_Store':
                continue
            try:
                data = loadmat("/".join([plot_dir, meth, file]))
            except NotImplementedError:
                #continue

                with h5py.File("/".join([plot_dir, meth, file]), 'r') as f:
                    data = {}
                    legend_name = re.sub(r"(wider_pr_info_)|(_easy_val\.mat)|(_medium_val\.mat)|(_hard_val\.mat)", '', file)
                    data['legend_name'] = [legend_name]
                    try:
                        data['pr_cruve'] = np.transpose(f['pr_cruve'][:])
                    except KeyError:
                        data['pr_cruve'] = np.transpose(f['pr_curve'][:])
            if 'easy' in file:
                AP = compute_AP(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0])
                axs_1.plot(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0], label=f"{data['legend_name'][0]}-{AP:.3f}", linewidth=4)
            elif 'medium' in file:
                AP = compute_AP(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0])
                axs_2.plot(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0], label=f"{data['legend_name'][0]}-{AP:.3f}", linewidth=4)
            else:
                AP = compute_AP(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0])
                axs_3.plot(data['pr_cruve'][:, 1], data['pr_cruve'][:, 0], label=f"{data['legend_name'][0]}-{AP:.3f}", linewidth=4)

    for ax in axs_1, axs_2, axs_3:
        ax.grid(True)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: float(t[0][-5:]), reverse=True)))
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

    axs_1.set_title("Easy")
    axs_2.set_title("Medium")
    axs_3.set_title("Hard")
    for fig in fig_1, fig_2, fig_3:
        fig.tight_layout()


    fig_1.savefig("results/Easy_WIDER.pdf")
    fig_2.savefig("results/Medium_WIDER.pdf")
    fig_3.savefig("results/Hard_WIDER.pdf")
    plt.show()
