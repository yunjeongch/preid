import torch
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import vonmises

def calculate_directions(keypoints_batch, img_width_batch):

    keypoints_normalized = keypoints_batch.clone()
    keypoints_normalized[:, :, 0:1] /= img_width_batch

    left_shoulder = keypoints_normalized[:, 5]
    right_shoulder = keypoints_normalized[:, 6]
    left_hip = keypoints_normalized[:, 11]
    right_hip = keypoints_normalized[:, 12]

    shoulders_direction = left_shoulder - right_shoulder
    hips_direction = left_hip - right_hip

    direction = torch.mean(torch.stack([shoulders_direction, hips_direction]), dim=0)
    return direction[:, 0]

def direction_similarity(dir1, dir2):
    sim = torch.abs(dir1 - dir2)
    return sim

def inferability(dir1, dir2, alpha):
    return torch.exp(-alpha * direction_similarity(dir1, dir2))

def calculate_weights(pred_keypoints, gt_keypoints, img_width, alpha = 0.5):
    pred_directions = calculate_directions(pred_keypoints, img_width)   # [batch]
    gt_directions = calculate_directions(gt_keypoints, img_width)   # [batch]
    inferabilities = inferability(pred_directions, gt_directions, alpha = alpha)
    return inferabilities

def front_back(keypoints, width):
    size_margin = width.squeeze() * 0.1

    left_shoulder_x = keypoints[:, 5, 0]
    right_shoulder_x = keypoints[:, 6, 0]
    left_hip_x = keypoints[:, 11, 0]
    right_hip_x = keypoints[:, 12, 0]

    # anchor가 front일 때
    front_mask = (left_shoulder_x > right_shoulder_x + size_margin) & (left_hip_x > right_hip_x + size_margin)
    # anchor가 back일 때
    back_mask = (left_shoulder_x + size_margin < right_shoulder_x) & (left_hip_x + size_margin < right_hip_x)
    # 나머지 경우 side
    # side_mask = ~(front_mask | back_mask)

    posture = torch.zeros(left_shoulder_x.shape[0], dtype = int)
    posture[front_mask] = 1
    posture[back_mask] = -1

    return posture

def discrete_weights(anchor, pos_or_neg, width):
    anc = front_back(anchor, width)
    pos_neg = front_back(pos_or_neg, width)
    # If anchor is front
    front_anchor_mask = (anc == 1)
    # If positive/negative is back
    front_pos_neg_mask = (pos_neg == -1)
    # If positive/negative is side
    # side_pos_neg_mask = ~(front_pos_neg_mask)

    # If anchor is back
    back_anchor_mask = (anc == -1)
    # If positive/negative is front
    back_pos_neg_mask = (pos_neg == 1)
    # If positive/negative is side or back
    # side_back_pos_neg_mask = ~(back_pos_neg_mask)

    weights = torch.ones_like(anc, dtype=torch.float32)
    weights[front_anchor_mask & front_pos_neg_mask] = 0.5
    weights[back_anchor_mask & back_pos_neg_mask] = 0.5

    return weights

def periodic_distribution_weights(query, counterpart, kappa1, kappa2, weight = 0.5, vis=False):
    """
    주어진 각도 데이터(query)에 대해 혼합 von Mises 분포의 확률 밀도를 계산하는 함수.
    Parameters:
    query (torch.Tensor): B개의 mu1에 해당하는 각도 데이터 (단위: degrees)
    counterpart (torch.Tensor): 혼합 분포의 확률 밀도를 계산할 각도 데이터 (단위: degrees)
    kappa1 (float): 첫 번째 von Mises 분포의 집중도
    kappa2 (float): 두 번째 von Mises 분포의 집중도
    weight (float): 첫 번째 분포의 가중치 (두 번째 분포의 가중치는 1 - weight)
    Returns:
    torch.Tensor: counterpart에 있는 각도에 대한 혼합 분포의 확률 밀도 함수 값 (단위: density)
    """
    B = query.shape[0]
    mu1 = query.cpu().numpy()
    mu2 = 360 - mu1
    mu2[mu2 > 180] -= 360
    mu1_rad = np.deg2rad(mu1)
    mu2_rad = np.deg2rad(mu2)
    counterpart_rad = np.deg2rad(counterpart.cpu().numpy())
    mixed_pdf_values = []
    min_value = 0.2
    for i in range(B):
        pdf1_value, pdf1_norm = vonmises.pdf(counterpart_rad[i], kappa1, loc=mu1_rad[i]), vonmises.pdf(mu1_rad[i], kappa1, loc=mu1_rad[i])
        pdf2_value, pdf2_norm = vonmises.pdf(counterpart_rad[i], kappa2, loc=mu2_rad[i]), vonmises.pdf(mu2_rad[i], kappa2, loc=mu2_rad[i])
        mixed_pdf_value = (1-min_value) * ((weight * pdf1_value + (1 - weight) * pdf2_value) / (weight * pdf1_norm + (1-weight) * pdf2_norm)) + min_value
        mixed_pdf_values.append(mixed_pdf_value)
        if vis and i%10 == 0:
            theta = np.linspace(-np.pi, np.pi, 360)
            pdf1 = vonmises.pdf(theta, kappa1, loc=mu1_rad[i])
            pdf2 = vonmises.pdf(theta, kappa2, loc=mu2_rad[i])
            mixed_pdf = weight * pdf1 + (1 - weight) * pdf2
            plt.figure(figsize=(10, 6))
            plt.plot(np.rad2deg(theta), pdf1, label=f'von Mises (mean={mu1[i]}°, kappa={kappa1})')
            plt.plot(np.rad2deg(theta), pdf2, label=f'von Mises (mean={mu2[i]}°, kappa={kappa2})')
            plt.plot(np.rad2deg(theta), mixed_pdf, label='Mixed von Mises', color='black', linestyle='--')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Density')
            plt.title('Mixed von Mises Distributions')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'mixed_von_mises_plot{i}.png')
    return torch.tensor(mixed_pdf_values)