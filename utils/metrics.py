
# import torch
# import torch.nn.functional as F
#
#
# def get_accuracy(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     corr = torch.sum(SR == GT)
#     tensor_size = SR.numel()
#     acc = (corr.float() / tensor_size).item()
#     return acc
#
#
# def get_sensitivity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TP = (SR & GT).sum().float()
#     FN = (~SR & GT).sum().float()
#     sensitivity = TP / (TP + FN + 1e-6)
#     return sensitivity.item()
#
#
# def get_specificity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TN = (~SR & ~GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     specificity = TN / (TN + FP + 1e-6)
#     return specificity.item()
#
#
# def get_precision(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TP = (SR & GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     precision = TP / (TP + FP + 1e-6)
#     return precision.item()
#
#
# def get_hd95(SR, GT, batch_size=2048):
#     """
#     计算Hausdorff距离的95%分位数（分批处理版本，减少内存占用）
#     SR: 二值化预测结果（布尔张量）
#     GT: 二值化真实标签（布尔张量）
#     batch_size: 每批处理的点数量（根据GPU内存调整）
#     """
#     # 提取前景像素坐标（格式：[y, x]）
#     sr_points = torch.nonzero(SR).float()  # 形状：[N, 2]
#     gt_points = torch.nonzero(GT).float()  # 形状：[M, 2]
#
#     # 处理空集情况
#     if len(sr_points) == 0 and len(gt_points) == 0:
#         return 0.0
#     if len(sr_points) == 0 or len(gt_points) == 0:
#         return float('inf')
#
#     # 分批计算所有预测点到最近真实点的距离
#     min_dist_sr = []
#     for i in range(0, len(sr_points), batch_size):
#         # 取当前批次的预测点
#         sr_batch = sr_points[i:i+batch_size]
#         # 计算当前批次到所有真实点的距离（矩阵大小：batch_size×M）
#         dist_batch = torch.cdist(sr_batch, gt_points, p=2)
#         # 取每个点的最近距离并保存
#         min_dist_sr.append(torch.min(dist_batch, dim=1).values)
#     # 合并所有批次的结果
#     min_dist_sr = torch.cat(min_dist_sr, dim=0)
#
#     # 分批计算所有真实点到最近预测点的距离
#     min_dist_gt = []
#     for i in range(0, len(gt_points), batch_size):
#         # 取当前批次的真实点
#         gt_batch = gt_points[i:i+batch_size]
#         # 计算当前批次到所有预测点的距离（矩阵大小：batch_size×N）
#         dist_batch = torch.cdist(gt_batch, sr_points, p=2)
#         # 取每个点的最近距离并保存
#         min_dist_gt.append(torch.min(dist_batch, dim=1).values)
#     # 合并所有批次的结果
#     min_dist_gt = torch.cat(min_dist_gt, dim=0)
#
#     # 计算95%分位数
#     hd95_sr = torch.quantile(min_dist_sr, q=0.95, interpolation='linear').item()
#     hd95_gt = torch.quantile(min_dist_gt, q=0.95, interpolation='linear').item()
#
#     return max(hd95_sr, hd95_gt)
#
#
# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output)  # 确保输出在0-1区间
#
#     # 二值化处理
#     output_ = output > 0.5
#     target_ = target > 0.5
#
#     # 计算IoU和Dice
#     intersection = (output_ & target_).sum().float()
#     union = (output_ | target_).sum().float()
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2.0 * iou) / (iou + 1.0)
#
#     # 计算其他指标
#     recall = get_sensitivity(output_, target_)
#     precision = get_precision(output_, target_)
#     specificity = get_specificity(output_, target_)
#     acc = get_accuracy(output_, target_)
#     F1 = 2 * recall * precision / (recall + precision + 1e-6)
#
#     # 计算HD95
#     hd95 = get_hd95(output_, target_)
#
#     return (iou.item(), dice.item(), recall, precision, F1, specificity, acc, hd95)

import torch
from medpy import metric  # 导入medpy的指标库


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5
    corr = torch.sum(SR == GT)
    tensor_size = SR.numel()
    acc = (corr.float() / tensor_size).item()
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5
    TP = (SR & GT).sum().float()
    FN = (~SR & GT).sum().float()
    sensitivity = TP / (TP + FN + 1e-6)
    return sensitivity.item()


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5
    TN = (~SR & ~GT).sum().float()
    FP = (SR & ~GT).sum().float()
    specificity = TN / (TN + FP + 1e-6)
    return specificity.item()


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5
    TP = (SR & GT).sum().float()
    FP = (SR & ~GT).sum().float()
    precision = TP / (TP + FP + 1e-6)
    return precision.item()

import numpy as np  # 导入numpy并取别名np
def get_hd95(SR, GT):
    """使用medpy库的binary.hd95计算HD95"""
    # 将PyTorch布尔张量转为numpy的uint8数组（medpy要求输入为0/1的数组）
    sr_np = SR.cpu().numpy().astype(np.uint8)
    gt_np = GT.cpu().numpy().astype(np.uint8)

    # 处理单通道图像的维度（medpy通常期望2D/3D无通道维度，若有通道则squeeze）
    sr_np = np.squeeze(sr_np)
    gt_np = np.squeeze(gt_np)

    # 检查前景是否存在
    sr_has_foreground = np.any(sr_np)
    gt_has_foreground = np.any(gt_np)

    # 情况1：两者都无前景（全背景）
    if not sr_has_foreground and not gt_has_foreground:
        return 0.0
    # 情况2：一方有前景，另一方无（完全不重叠）
    if not sr_has_foreground or not gt_has_foreground:
        return float('inf')
    # 情况3：两者都有前景，调用库函数计算hd95
    return metric.binary.hd95(sr_np, gt_np)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output)  # 确保输出在0-1区间

    # 二值化处理
    output_ = output > 0.5
    target_ = target > 0.5

    # 计算IoU和Dice
    intersection = (output_ & target_).sum().float()
    union = (output_ | target_).sum().float()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * iou) / (iou + 1.0)

    # 计算其他指标
    recall = get_sensitivity(output_, target_)
    precision = get_precision(output_, target_)
    specificity = get_specificity(output_, target_)
    acc = get_accuracy(output_, target_)
    F1 = 2 * recall * precision / (recall + precision + 1e-6)

    # 计算HD95（调用整合medpy的函数）
    hd95 = get_hd95(output_, target_)

    return (iou.item(), dice.item(), recall, precision, F1, specificity, acc, hd95)


# import torch
# import torch.nn.functional as F
#
#
# def get_accuracy(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     corr = torch.sum(SR == GT)
#     tensor_size = SR.numel()
#     acc = (corr.float() / tensor_size).item()
#     return acc
#
#
# def get_sensitivity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TP = (SR & GT).sum().float()
#     FN = (~SR & GT).sum().float()
#     sensitivity = TP / (TP + FN + 1e-6)
#     return sensitivity.item()
#
#
# def get_specificity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TN = (~SR & ~GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     specificity = TN / (TN + FP + 1e-6)
#     return specificity.item()
#
#
# def get_precision(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5
#     TP = (SR & GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     precision = TP / (TP + FP + 1e-6)
#     return precision.item()
#
#
# def get_hd95(output, target):
#     """
#     计算Hausdorff距离的95th百分位数
#     output和target为二值布尔张量 (B, H, W) 或 (H, W)
#     """
#     # 确保输入是2D或3D
#     if output.dim() > 3:
#         # 如果维度太高，取第一个通道或展平批次维度
#         output = output.squeeze()
#     if target.dim() > 3:
#         target = target.squeeze()
#
#     # 如果是批次数据，递归处理每个样本
#     if output.dim() == 3:
#         hd95_values = []
#         for i in range(output.shape[0]):
#             hd95_val = get_hd95_single(output[i], target[i])
#             hd95_values.append(hd95_val)
#         return sum(hd95_values) / len(hd95_values)
#     else:
#         return get_hd95_single(output, target)
#
#
# def get_hd95_single(output, target):
#     """
#     处理单张图像的HD95计算
#     """
#     device = output.device
#
#     # 确保是2D张量
#     if output.dim() > 2:
#         output = output.squeeze()
#     if target.dim() > 2:
#         target = target.squeeze()
#
#     # 3x3结构元素用于腐蚀操作
#     kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
#     padding = 1
#
#     # 提取output的边界点
#     output_float = output.float()
#     if output_float.dim() == 2:
#         output_float = output_float.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#     elif output_float.dim() == 3:
#         output_float = output_float.unsqueeze(1)  # (B, 1, H, W) 但这里应该是单张图
#
#     eroded_output = F.conv2d(output_float, kernel, padding=padding, stride=1) == 9.0
#     eroded_output = eroded_output.squeeze().bool()
#     boundary_output = output & ~eroded_output
#     coords_output = torch.nonzero(boundary_output, as_tuple=False).float()
#
#     # 提取target的边界点
#     target_float = target.float()
#     if target_float.dim() == 2:
#         target_float = target_float.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#     elif target_float.dim() == 3:
#         target_float = target_float.unsqueeze(1)
#
#     eroded_target = F.conv2d(target_float, kernel, padding=padding, stride=1) == 9.0
#     eroded_target = eroded_target.squeeze().bool()
#     boundary_target = target & ~eroded_target
#     coords_target = torch.nonzero(boundary_target, as_tuple=False).float()
#
#     # 处理空边界的特殊情况
#     if coords_output.numel() == 0 and coords_target.numel() == 0:
#         return 0.0
#     if coords_output.numel() == 0 or coords_target.numel() == 0:
#         H, W = output.shape
#         max_dist = torch.sqrt(torch.tensor(H ** 2 + W ** 2, dtype=torch.float32, device=device))
#         return max_dist.item()
#
#     # 计算双向最近距离
#     dists_A = torch.cdist(coords_output, coords_target, p=2)
#     min_dists_A = dists_A.min(dim=1).values
#     dists_B = torch.cdist(coords_target, coords_output, p=2)
#     min_dists_B = dists_B.min(dim=1).values
#
#     # 合并距离并计算95%分位数
#     all_dists = torch.cat([min_dists_A, min_dists_B])
#     hd95 = torch.quantile(all_dists, 0.95, interpolation='linear').item()
#
#     return hd95
#
#
# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output)
#
#     # 确保输出和目标的维度一致
#     output_flat = output
#     target_flat = target
#
#     # 如果维度不匹配，调整维度
#     if output_flat.dim() != target_flat.dim():
#         # 尝试自动调整维度
#         if output_flat.dim() == 4 and target_flat.dim() == 3:
#             target_flat = target_flat.unsqueeze(1)
#         elif output_flat.dim() == 3 and target_flat.dim() == 4:
#             output_flat = output_flat.unsqueeze(1)
#
#     # 二值化分割结果和目标掩码
#     output_ = output_flat > 0.5
#     target_ = target_flat > 0.5
#
#     # 如果还是高维数据，选择第一个通道
#     if output_.dim() > 3:
#         output_ = output_[:, 0] if output_.shape[1] > 1 else output_.squeeze(1)
#     if target_.dim() > 3:
#         target_ = target_[:, 0] if target_.shape[1] > 1 else target_.squeeze(1)
#
#     # 确保是2D或3D张量
#     output_ = output_.squeeze()
#     target_ = target_.squeeze()
#
#     # 如果是3D（批次数据），取第一个或者计算平均
#     if output_.dim() == 3:
#         # 使用第一个样本计算指标
#         output_single = output_[0] if output_.shape[0] > 0 else output_
#         target_single = target_[0] if target_.shape[0] > 0 else target_
#     else:
#         output_single = output_
#         target_single = target_
#
#     # 计算IoU和Dice
#     intersection = (output_single & target_single).sum().float()
#     union = (output_single | target_single).sum().float()
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2.0 * iou) / (iou + 1.0)
#
#     # 计算其他指标（使用原始的二值化结果）
#     recall = get_sensitivity(output_single, target_single)
#     precision = get_precision(output_single, target_single)
#     specificity = get_specificity(output_single, target_single)
#     acc = get_accuracy(output_single, target_single)
#     F1 = 2 * recall * precision / (recall + precision + 1e-6)
#
#     # 计算HD95
#     hd95 = get_hd95_single(output_single, target_single)
#
#     return (iou.item(), dice.item(), recall, precision, F1, specificity, acc, hd95)















# import torch
#
#
# def get_accuracy(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5  # 修正GT处理
#     corr = torch.sum(SR == GT)
#     tensor_size = SR.numel()
#     acc = (corr.float() / tensor_size).item()
#     return acc
#
#
# def get_sensitivity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5  # 修正GT处理
#     TP = (SR & GT).sum().float()
#     FN = (~SR & GT).sum().float()
#     sensitivity = TP / (TP + FN + 1e-6)
#     return sensitivity.item()
#
#
# def get_specificity(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5  # 修正GT处理
#     TN = (~SR & ~GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     specificity = TN / (TN + FP + 1e-6)
#     return specificity.item()
#
#
# def get_precision(SR, GT, threshold=0.5):
#     SR = SR > threshold
#     GT = GT > 0.5  # 修正GT处理
#     TP = (SR & GT).sum().float()
#     FP = (SR & ~GT).sum().float()
#     precision = TP / (TP + FP + 1e-6)
#     return precision.item()
#
#
# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output)  # 确保输出在0-1之间
#
#     output_ = output > 0.5
#     target_ = target > 0.5
#
#     intersection = (output_ & target_).sum().float()
#     union = (output_ | target_).sum().float()
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2.0 * iou) / (iou + 1.0)  # 维持原Dice计算方式
#
#     # 计算其他指标时直接使用二值化的output_和target_
#     recall = get_sensitivity(output_, target_)
#     precision = get_precision(output_, target_)
#     specificity = get_specificity(output_, target_)
#     acc = get_accuracy(output_, target_)
#     F1 = 2 * recall * precision / (recall + precision + 1e-6)
#
#     return (iou.item(), dice.item(), recall, precision, F1, specificity, acc)
#
# # def iou_score(output_tuple, target, threshold=0.5):
# #     smooth = 1e-5
# #     iou_list = []
# #     dice_list = []
# #     recall_list = []
# #     precision_list = []
# #     f1_list = []
# #     specificity_list = []
# #     acc_list = []
# #
# #     # 遍历元组中的每个输出
# #     for output in output_tuple:
# #         # 确保输出是张量
# #         if not torch.is_tensor(output):
# #             raise TypeError("Each element in the output tuple must be a tensor.")
# #
# #         # 应用 sigmoid 和阈值处理
# #         output_process = torch.sigmoid(output)  # 确保输出在0-1之间
# #         output_ = output_process > threshold
# #         target_ = target > 0.5  # 修正GT处理
# #
# #         # 计算IoU和Dice
# #         intersection = (output_ & target_).sum().float()
# #         union = (output_ | target_).sum().float()
# #         iou = (intersection + smooth) / (union + smooth)
# #         dice = (2.0 * iou) / (iou + 1.0)  # 维持原Dice计算方式
# #
# #         # 计算其他指标
# #         TP = (output_ & target_).sum().float()
# #         FN = (~output_ & target_).sum().float()
# #         FP = (output_ & ~target_).sum().float()
# #         TN = (~output_ & ~target_).sum().float()
# #
# #         recall = TP / (TP + FN + 1e-6)
# #         precision = TP / (TP + FP + 1e-6)
# #         F1 = 2 * recall * precision / (recall + precision + 1e-6) if (recall + precision) != 0 else 0.0
# #         specificity = TN / (TN + FP + 1e-6)
# #         acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
# #
# #         # 将结果添加到列表中
# #         iou_list.append(iou.item())
# #         dice_list.append(dice.item())
# #         recall_list.append(recall.item())
# #         precision_list.append(precision.item())
# #         f1_list.append(F1)
# #         specificity_list.append(specificity.item())
# #         acc_list.append(acc.item())
# #
# #     # 计算每个指标的平均值
# #     avg_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0.0
# #     avg_dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0
# #     avg_recall = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0.0
# #     avg_precision = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0.0
# #     avg_specificity = sum(specificity_list) / len(specificity_list) if len(specificity_list) > 0 else 0.0
# #     avg_acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0.0
# #     avg_f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0.0
# #
# #     return (avg_iou, avg_dice, avg_recall, avg_precision, avg_f1, avg_specificity, avg_acc)