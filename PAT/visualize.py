import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import argparse
import json
import logging
from data.data_utils import read_image


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dset_name', type = str)
parser.add_argument('--pred_path', type = str)

args = parser.parse_args()

logger.setLevel(logging.INFO)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def plot_sample(save_dir, row):
    qidx, qid, ginfos, qpath, gpaths = row['query_index'], row['query_pid'], row['top5_gallery'], row['query_path'], row['top5_gallery_path']
    fig_img = plt.figure(figsize=(20,6))
    gs = GridSpec(ncols=6, nrows=1, figure=fig_img)
    fig_img.suptitle(f"Query index: {qidx}; Query pid: {qid}")

    for i in range(6):
        ax = fig_img.add_subplot(gs[0, i % 6])
        
        if i == 0:
            ax.title.set_text(f'\n\nQuery pid: {qid}')
            ax.imshow(read_image(qpath))
        else:
            ax.title.set_text(f'\nTop {i} pid: {ginfos[i-1][0]}')
            ax.imshow(read_image(gpaths[i-1]))

        color = 'black' if i==0 else ('blue' if qid == ginfos[i-1][0] else 'red')
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    fig_img.tight_layout()
    # plt.show()
    file_path = os.path.join(save_dir, f"{row['query_index']}_{row['query_pid']}.png")
    fig_img.savefig(file_path)
    plt.close()
    return

def main(args):
    files = os.listdir(args.pred_path)
    for file in files:
        pred = pd.DataFrame(load_jsonl(os.path.join(args.pred_path,file)))
        save_dir = f'samples/{args.dset_name}/{file.split("_samples")[0]}/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for _, row in tqdm(pred.iterrows()):
            plot_sample(save_dir, row)


if __name__ == '__main__':
    main(args)