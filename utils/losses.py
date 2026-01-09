import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss', 'DeepSupervisionBCEDiceLoss', 'compute_kl_loss']

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

#
# class BCEIoULoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         # 计算二元交叉熵损失
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#
#         # 应用Sigmoid激活函数将logits转换为概率
#         input = torch.sigmoid(input)
#
#         num = target.size(0)
#         # 将输入和目标展平为2D张量（batch_size, num_pixels）
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#
#         # 计算交集和并集
#         intersection = (input * target).sum(1)  # 每个样本的交集
#         total = (input + target).sum(1)  # 每个样本的输入和目标和
#         union = total - intersection  # 并集 = 总和 - 交集
#
#         # 计算IoU： (交集 + 平滑) / (并集 + 平滑)
#         iou = (intersection + smooth) / (union + smooth)
#         # 计算IoU Loss：1 - IoU，然后在整个批次上平均
#         iou_loss = 1 - iou
#         iou_loss = iou_loss.sum() / num  # 批次平均IoU损失
#
#         # 组合损失：BCE + IoU Loss
#         return 0.5 * bce + iou_loss

class DeepSupervisionBCEDiceLoss(nn.Module):
    def __init__(self, weights=[1, 1, 1, 1, 1]):
    # def __init__(self, weights = [1.0, 0.8, 0.6, 0.4, 0.2]):
    # def __init__(self, weights=[0.2, 0.4, 0.6, 0.8, 1.0]):
        super().__init__()
        self.base_loss = BCEDiceLoss()
        self.weights = weights

    def forward(self, outputs, targets):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)

        total_loss = 0
        for output, weight in zip(outputs, self.weights):
            loss = self.base_loss(output, targets)
            total_loss += weight * loss
        # return total_loss / len(self.weights)  # 平均加权损失
        return total_loss









def compute_kl_loss(p, q):
    #用于计算两个概率分布之间的Kullback-Leibler散度（KL散度）
    #KL散度是衡量两个概率分布之间差异的一个重要指标
    #p: 第一个概率分布的预测值，通常是模型的输出
    #q: 第二个概率分布的目标值，通常是实际标签的分布。
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')#计算p到q的kl散度
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),#计算q到p的kl散度
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()#平均损失
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
#最终的损失是 p_loss 和 q_loss 的平均值


""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.seg_loss = BCEDiceLoss()
        # self.boundary_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha  # 平衡系数

    def forward(self, seg_pred, boundary_preds, seg_gt, boundary_gt):
        # 分割损失
        loss_seg = self.seg_loss(seg_pred, seg_gt)
        # 多尺度边界损失
        loss_boundary = 0
        for pred in boundary_preds:
            pred = F.interpolate(pred, boundary_gt.shape[2:], mode='bilinear')
            loss_boundary += F.binary_cross_entropy_with_logits(pred, boundary_gt)

        return loss_seg + self.alpha * loss_boundary



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from scipy.ndimage import distance_transform_edt as edt
#
# try:
#     from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
# except ImportError:
#     pass
#
# # __all__ = ['BCEDiceLoss', 'LovaszHingeLoss']
#
# class HausdorffDTLoss(nn.Module):
#     """Binary Hausdorff loss based on distance transform"""
#
#     def __init__(self, alpha=2.0, **kwargs):
#         super(HausdorffDTLoss, self).__init__()
#         self.alpha = alpha
#         self.min_loss = float('inf')
#         self.max_loss = float('-inf')
#
#     @torch.no_grad()
#     def update_min_max_loss(self, loss):
#         self.min_loss = min(self.min_loss, loss)
#         self.max_loss = max(self.max_loss, loss)
#
#     @torch.no_grad()
#     def normalize_loss(self, loss):
#         return (loss - self.min_loss) / (self.max_loss - self.min_loss + 1e-8)
#
#     @torch.no_grad()
#     def distance_field(self, img: np.ndarray) -> np.ndarray:
#         field = np.zeros_like(img)
#
#         for batch in range(len(img)):
#             fg_mask = img[batch] > 0.5
#
#             if fg_mask.any():
#                 bg_mask = ~fg_mask
#
#                 fg_dist = edt(fg_mask)
#                 bg_dist = edt(bg_mask)
#
#                 field[batch] = fg_dist + bg_dist
#
#         return field
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
#         """
#         Uses one binary channel: 1 - fg, 0 - bg
#         pred: (b, 1, x, y) or (b, 1, x, y, z)
#         target: (b, 1, x, y) or (b, 1, x, y, z)
#         """
#         assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
#         assert target.dim() == pred.dim(), "Target should have the same dimension as prediction"
#
#         device = pred.device  # Get the device of the input tensor
#
#         # Apply threshold to pred and target
#         pred_binary = (pred >= 0.5).float()  # Convert values >= 0.5 to 1, and < 0.5 to 0
#         target_binary = (target >= 0.5).float()  # Convert values >= 0.5 to 1, and < 0.5 to 0
#
#         # Compute distance transforms
#         pred_dt = torch.from_numpy(self.distance_field(pred_binary.cpu().numpy())).float().to(device)  # Move to the same device
#         target_dt = torch.from_numpy(self.distance_field(target_binary.cpu().numpy())).float().to(device)  # Move to the same device
#
#         # Compute Hausdorff distance loss
#         pred_error = (pred_binary - target_binary) ** 2
#         distance = pred_dt ** self.alpha + target_dt ** self.alpha
#
#         dt_field = pred_error * distance
#         class_loss = dt_field.mean()
#
#         self.update_min_max_loss(class_loss.item())
#         normalized_loss = self.normalize_loss(class_loss)
#
#         if debug:
#             return (
#                 normalized_loss.cpu().numpy(),
#                 (
#                     dt_field.cpu().numpy()[0, 0],
#                     pred_error.cpu().numpy()[0, 0],
#                     distance.cpu().numpy()[0, 0],
#                     pred_dt.cpu().numpy()[0, 0],
#                     target_dt.cpu().numpy()[0, 0],
#                 ),
#             )
#         else:
#             return normalized_loss
# #该损失函数通过计算预测和目标图像的距离变换，并结合误差平方来衡量预测结果与真实标签之间的差异。
#
#
# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hd_loss = HausdorffDTLoss(alpha=2.0)
#
#     def forward(self, input, target):
#
#         hd_loss_value = self.hd_loss(input, target)
#
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#
#
#
#         # Combine the losses
#         return 0.5 * bce + dice + 0.5 * hd_loss_value
#
#
# class LovaszHingeLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         input = input.squeeze(1)
#         target = target.squeeze(1)
#         loss = lovasz_hinge(input, target, per_image=True)
#
#         return loss