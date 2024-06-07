from re import T
from time import time
import torch
import numpy as np
import os
from utils.reranking import re_ranking
from utils.file_io import save_jsonl
import matplotlib.pyplot as plt


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, save_dir, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    wrong_predictions = []
    wrong_at_all = []
    pred = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_path = q_paths[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        top5_gallery_info = [(int(g_pids[i]), int(g_camids[i])) for i in order[keep][:5]]
        top5_gallery_paths = [g_paths[i] for i in order[keep][:5]]
        
        pred.append({
                'query_index': int(q_idx),
                'query_pid': int(q_pid),
                'query_path': q_path,
                'query_camid': int(q_camid),
                'top5_gallery': top5_gallery_info,
                'top5_gallery_path': top5_gallery_paths,
                'correct': bool(cmc[0] == 1)
            })
        
        # if q_pid != g_pids[order[0]]:
        #     wrong_at_all.append({
        #         'query_index': int(q_idx),
        #         'query_pid': int(q_pid),
        #         'query_camid': int(q_camid),
        #         'query_path': q_path,
        #         'top5_gallery': top5_gallery_info,
        #         'top5_gallery_path': top5_gallery_paths
        #     })
        # elif cmc[0] != 1:
        #     wrong_predictions.append({
        #         'query_index': int(q_idx),
        #         'query_pid': int(q_pid),
        #         'query_path': q_path,
        #         'query_camid': int(q_camid),
        #         'top5_gallery': top5_gallery_info,
        #         'top5_gallery_path': top5_gallery_paths
        #     })

    # if save_dir and wrong_predictions:
    #     save_jsonl(wrong_predictions, save_dir)
    
    # if save_dir and wrong_at_all:
    #     save_jsonl(wrong_at_all, save_dir.replace('wrong', 'wrong_at_all'))
            
    if save_dir:
        save_jsonl(pred, save_dir)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpaths = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, imgpath = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.imgpaths.extend(imgpath)

    def compute(self, save_dir = ''):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_paths = np.asarray(self.imgpaths[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_paths = np.asarray(self.imgpaths[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_paths, g_paths, save_dir)
        
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

class R1_mAP_with_pose(R1_mAP_eval):
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_with_pose, self).__init__(self, num_query, max_rank, feat_norm)
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpaths = []
        self.directions = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, imgpath, dirs = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.imgpaths.extend(imgpath)
        self.directions.extend(dirs)
    
    def plot_direction_differences(self, direction_differences, output_file):
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
        for i in range(len(direction_differences[:10])):
            ax.plot(range(1, direction_differences[i].shape[0] + 1), direction_differences[i], marker = 'o', alpha = 0.3)
        plt.xlabel('Rank')
        plt.ylabel('Absolute Direction Difference')
        plt.title('Absolute Direction Difference by Relative Rank')
        plt.grid(True)
        plt.savefig(output_file)
        print('Successfully saved the plot')
        return
    
    def plot_ranks(self, ranks, output_file):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ranks) + 1), ranks, marker = 'o', alpha = 1)
        plt.xlabel('Rank')
        plt.ylabel('Absolute Direction Difference Ratio')
        plt.title('Absolute Direction Difference by Relative Rank')
        plt.grid(True)
        plt.savefig(output_file)
        print('Successfully saved the plot')
        return


    def relative_rank_of_GT(self, save_dir = '', max_rank = 50):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            # print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_paths = np.asarray(self.imgpaths[:self.num_query])
        q_directions = np.asarray(self.directions[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_paths = np.asarray(self.imgpaths[self.num_query:])
        g_directions = np.asarray(self.directions[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        rankings = []
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            # print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        
        num_q, num_g = distmat.shape
        # distmat g
        #    q    1 3 2 4
        #         4 1 2 3
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        #  0 2 1 3
        #  1 2 3 0
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        direction_differences = []
        ranks = [0, 0, 0, 0, 0]
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            q_path = q_paths[q_idx]
            q_direction = q_directions[q_idx]
        
            order = indices[q_idx]  # select one row
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                continue

            gt_indices = order[np.argwhere(matches[q_idx][keep]==1).squeeze(axis = 1)]
            gallery_directions = g_directions[gt_indices]
            direction_diff = np.abs(gallery_directions - q_direction)
            direction_differences.append(direction_diff)
            for i in range(len(direction_diff)):
                ranks[i] += direction_diff[i]

            rankings.append({
                'query_index': int(q_idx),
                'query_path': q_path,
                'query_direction': int(q_direction),
                'gt_directions': gallery_directions.tolist(),
                'direction_differences': [int(d) for d in direction_diff],
                'gt_path': g_paths[gt_indices].tolist()
            })

        
        save_jsonl(rankings, save_dir + '.jsonl')
        self.plot_direction_differences(direction_differences, save_dir + '_rank_pose.png')
        ranks = [r / len(direction_differences) for r in ranks]
        self.plot_ranks(ranks, save_dir + '_ratio.png')
        return
