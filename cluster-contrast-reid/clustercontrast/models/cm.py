import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)        # 初始化一个默认字典，用于存储每个类别的特征
        for instance_feature, index in zip(inputs, targets.tolist()):   # 遍历输入和目标标签，将每个类别的特征存储到字典中
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():    # 遍历每个类别的特征
            distances = []
            for feature in features:    # 计算每个特征与存储的特征之间的距离
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))      # 找到距离最小的特征的索引
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]        # 使用动量更新存储的特征
            ctx.features[index] /= ctx.features[index].norm()   #归一化

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features,momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, text_output, targets, features, text_center, momentum):
        ctx.features = features
        ctx.text_center = text_center
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, text_output, targets)

        # *** 关键修改：detach缓冲区，避免计算图累积 ***
        outputs = inputs.mm(ctx.features.detach().t())
        outputs_2 = text_output.mm(ctx.text_center.detach().t())
        return outputs, outputs_2

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs, text_output, targets = ctx.saved_tensors

        grad_inputs = None
        grad_text_output = None
        grad_targets = None
        grad_features = None
        grad_text_center = None
        grad_momentum = None

        # 计算梯度（使用detach后的缓冲区）
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs1.mm(ctx.features.detach())

        if ctx.needs_input_grad[1]:
            grad_text_output = grad_outputs2.mm(ctx.text_center.detach())

        # *** 关键修改：在no_grad下更新，并确保结果detach ***
        with torch.no_grad():
            for x, y in zip(inputs, targets):
                # 更新并立即detach，切断计算图
                ctx.features[y] = (ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x).detach()
                ctx.features[y] /= ctx.features[y].norm()

            for x, y in zip(text_output, targets):
                ctx.text_center[y] = (ctx.momentum * ctx.text_center[y] + (1. - ctx.momentum) * x).detach()
                ctx.text_center[y] /= ctx.text_center[y].norm()

        return grad_inputs, grad_text_output, grad_targets, grad_features, grad_text_center, grad_momentum


def cm(inputs, text_output, indexes, features, text_center, momentum=0.5):
    return CM.apply(inputs, text_output, indexes, features, text_center, momentum)


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        # 注册缓冲区
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('text_center', torch.zeros(num_samples, num_features))

    def forward(self, inputs, text_output, targets):
        # 归一化输入
        inputs = F.normalize(inputs, dim=1).to(inputs.device)
        text_output = F.normalize(text_output, dim=1).to(inputs.device)

        # *** 关键修改：在forward前确保缓冲区是detached的 ***
        # 这一步非常重要，防止缓冲区保留上一次迭代的计算图
        self.features = self.features.detach()
        self.text_center = self.text_center.detach()

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
            outputs_2 = None
        else:
            outputs, outputs_2 = cm(inputs, text_output, targets,
                                    self.features, self.text_center, self.momentum)

        # 温度缩放
        outputs = outputs / self.temp
        outputs_2 = outputs_2 / self.temp

        # 计算损失
        loss_img = F.cross_entropy(outputs, targets)
        loss_text = F.cross_entropy(outputs_2, targets)
        total_loss = (loss_img + loss_text)

        return total_loss, loss_img, loss_text
