import torch

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