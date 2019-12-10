from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2
import math
import cmath
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cal_mse(im1, im2):
    mse = (((im1 * 1.0 - im2 * 1.0).abs()) ** 2).mean()
    # mse = np.mean((im1 / 1.0 - im2 / 1.0) ** 2)
    return mse


def MDSI(RefImg, DistImg, combMethod='sum'):
    # RefImg = RefImg.cpu().detach().numpy()
    # DistImg = DistImg.cpu().detach().numpy()
    RefImg1 = RefImg.clone()
    RefImg1.mul(255).byte()
    RefImg1 = RefImg1.cpu().detach().numpy().transpose((1, 2, 0))
    DistImg1 = DistImg.clone()
    DistImg1.mul(255).byte()
    DistImg1 = DistImg1.cpu().detach().numpy().transpose((1, 2, 0))

    # print(RefImg.shape)
    # cv2.imshow('image', RefImg)
    # cv2.waitKey(0)
    B_ref, G_ref, R_ref = cv2.split(RefImg1)  # 分离RGB颜色通道
    B_dist, G_dist, R_dist = cv2.split(DistImg1)

    C1 = 140
    C2 = 55
    C3 = 550
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 3
    dy = np.transpose(dx)
    rows, cols = np.shape(R_ref)
    minDimension = min(rows, cols)
    f = max(1, round(minDimension / 256))

    # 跳行（列）提取矩阵的元素，参考numpy advanced indexing [https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html]
    row_index, col_index = np.meshgrid([i for i in range(0, rows, f)], [i for i in range(0, cols, f)])
    row_index = np.transpose(row_index)
    col_index = np.transpose(col_index)

    aveR1 = cv2.blur(R_ref, (f, f))  # 当卷积核对称时，卷积等于滤波
    aveR2 = cv2.blur(R_dist, (f, f))
    R1 = aveR1[row_index, col_index]
    R2 = aveR2[row_index, col_index]

    aveG1 = cv2.blur(G_ref, (f, f))
    aveG2 = cv2.blur(G_dist, (f, f))
    G1 = aveG1[row_index, col_index]
    G2 = aveG2[row_index, col_index]

    aveB1 = cv2.blur(B_ref, (f, f))
    aveB2 = cv2.blur(B_dist, (f, f))
    B1 = aveB1[row_index, col_index]
    B2 = aveB2[row_index, col_index]

    # Luminance
    L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
    L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2
    F = 0.5 * (L1 + L2)  # Fusion

    # Opponent color space
    H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1
    H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2
    M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1
    M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2

    # Gradient magnitudes

    IxL1 = cv2.filter2D(L1, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)  # 当卷积核不对称时，需要将卷积核旋转90度
    IyL1 = cv2.filter2D(L1, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
    gR = np.sqrt(IxL1 ** 2 + IyL1 ** 2)

    IxL2 = cv2.filter2D(L2, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)
    IyL2 = cv2.filter2D(L2, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
    gD = np.sqrt(IxL2 ** 2 + IyL2 ** 2)

    IxF = cv2.filter2D(F, -1, np.flip(dx, -1), borderType=cv2.BORDER_CONSTANT)
    IyF = cv2.filter2D(F, -1, np.flip(dy, -1), borderType=cv2.BORDER_CONSTANT)
    gF = np.sqrt(IxF ** 2 + IyF ** 2)

    # Gradient Similarity(GS)
    GS12 = (2 * gR * gD + C1) / (gR ** 2 + gD ** 2 + C1)  # GS of R and D
    GS13 = (2 * gR * gF + C2) / (gR ** 2 + gF ** 2 + C2)
    GS23 = (2 * gD * gF + C2) / (gD ** 2 + gF ** 2 + C2)
    GS_HSV = GS12 + GS23 - GS13

    # Chromaticity Similarity(CS)
    CS = (2 * (H1 * H2 + M1 * M2) + C3) / (H1 ** 2 + H2 ** 2 + M1 ** 2 + M2 ** 2 + C3)
    # cv2.imshow('CS', CS)

    GCS = CS
    if combMethod == 'sum':
        alpha = 0.6
        GCS = alpha * GS_HSV + (1 - alpha) * CS
    elif combMethod is 'mult':
        gamma = 0.2
        beta = 0.1
        GCS = (GS_HSV ** gamma) * (CS ** beta)
    # cv2.imshow("GCS", GCS)
    # cv2.waitKey()
    cv2.destroyAllWindows()
    flatten = np.reshape(GCS, (-1, 1))
    temp = np.zeros(flatten.shape, dtype=np.complex64)
    for i, ele in enumerate(flatten):
        temp[i] = cmath.sqrt(ele)  # 程序里有两次开平方，第一次开平方在这里，用于构造numpy complex型的矩阵，cmath.sqrt()允许对负数开方，但是貌似一次只能操作一个数
    score = mad(temp ** 0.5) ** 0.25
    # score = torch.from_numpy(score)
    return score  # 第二次开平方及计算完mad之后的0.25次方


def mad(vec):
    """
    Mean Absolute Deviation
    :param array: numpy array
    :return: float
    """
    return np.mean(np.abs(vec - np.mean(vec)))


def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, with_mask, poses, poses_inv,
                                    max_scales, with_ssim=True, rotation_mode='euler', padding_mode='zeros'):
    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):
            b, _, h, w = tgt_depth[s].size()
            downscale = tgt_img.size(2) / h

            tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            intrinsic_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

            photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth[s],
                                                                ref_depth[s], pose, with_mask, intrinsic_scaled,
                                                                with_ssim)
            photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth[s],
                                                                tgt_depth[s], pose_inv, with_mask, intrinsic_scaled,
                                                                with_ssim)

            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, with_mask, intrinsic, with_ssim,
                          rotation_mode='euler', padding_mode='zeros'):
    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth,
                                                                                pose, intrinsic, rotation_mode,
                                                                                padding_mode)
    reconstruction_loss = 0
    diff_img = (tgt_img - ref_img_warped).abs() * valid_mask
    # mse_loss = cal_mse(tgt_img * valid_mask, ref_img_warped * valid_mask)

    if with_mask:
        # weight_mask = (1 - diff_depth) * valid_mask
        diff_img = diff_img

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth).abs()).clamp(0,
                                                                                                             1) * valid_mask
    reconstruction_loss = reconstruction_loss + mean_on_mask(diff_img, valid_mask)
    # reconstruction_loss = reconstruction_loss + mse_loss

    if with_ssim:
        mdsi_loss = 0
        for i in range(4):
            mdsi_loss += MDSI(tgt_img[i, :, :, :], ref_img_warped[i, :, :, :], combMethod='sum')
        mdsi_loss *= 0.25
        reconstruction_loss = 0.15 * reconstruction_loss + 0.85 * mdsi_loss

    # compute loss
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum()
    return mean_value


def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 4.0

    return loss


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs, max_scales=1):
    loss = edge_aware_smoothness_loss(tgt_depth, tgt_img, max_scales)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += edge_aware_smoothness_loss(ref_depth, ref_img, max_scales)

    return loss


@torch.no_grad()
def compute_errors(gt, pred):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    crop_mask = gt[0] != gt[0]
    y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
    x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
    crop_mask[y1:y2, x1:x2] = 1
    max_depth = 80

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
