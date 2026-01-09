import argparse


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):#用于重置所有实例变量。
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):#val 表示当前要更新的数值，n 是更新的次数，默认为 1。
        self.val = val
        #将传入的当前值 val 赋值给实例变量 self.val。这样，self.val 始终保持了最新的输入值。
        self.sum += val * n
        #计算总和 self.sum，将当前值 val 乘以更新次数 n 后加到 self.sum 上。这允许该方法在多个样本中进行累加
        self.count += n
        #将更新次数 n 加到计数器 self.count 上
        self.avg = self.sum / self.count
        #来计算当前的平均值
