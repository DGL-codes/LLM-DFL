"""损失函数模块：用于去中心化联邦遗忘的特殊损失函数。

包含：
- GradAscentLoss: 梯度上升损失，用于 D-Oblivionis 方法
- UCELoss: 遗忘交叉熵损失，用于 D-FedOSD 方法
"""
import torch
import torch.nn.functional as F
from typing import Optional


class GradAscentLoss:
    """梯度上升损失函数。
    
    通过取负交叉熵损失实现梯度上升，使模型"忘记"遗忘数据。
    
    原理：
    - 标准训练：最小化 loss = CE(pred, target)
    - 梯度上升：最大化 loss = -CE(pred, target)
    
    使用方式：
    ```python
    loss_fn = GradAscentLoss()
    loss = loss_fn(logits, labels)
    loss.backward()  # 梯度方向会使模型远离正确预测
    ```
    
    注意：为了数值稳定性，loss 会被裁剪到 [-max_loss, max_loss] 范围内
    """
    
    def __init__(self, ignore_index: int = -100, reduction: str = 'mean', max_loss: float = 100.0):
        """初始化 GradAscentLoss。
        
        Args:
            ignore_index: 忽略的标签索引（默认 -100）
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
            max_loss: 损失裁剪的最大绝对值（默认 100.0，防止数值溢出）
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.max_loss = max_loss
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算梯度上升损失。
        
        Args:
            logits: 模型输出 logits，形状 (batch_size, num_classes) 或 (batch_size, seq_len, vocab_size)
            labels: 目标标签，形状 (batch_size,) 或 (batch_size, seq_len)
        
        Returns:
            负的交叉熵损失（用于梯度上升），裁剪到 [-max_loss, max_loss]
        """
        # 计算标准交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )
        
        # 取负实现梯度上升，并裁剪防止数值溢出。
        # 阈值不能太低；否则 hard clamp 会让高 CE 样本的梯度变成 0，
        # 在后门遗忘这类高置信错误样本上会削弱 D-Oblivionis 的遗忘阶段。
        grad_ascent_loss = -ce_loss
        
        # 裁剪损失值，防止梯度爆炸
        grad_ascent_loss = torch.clamp(grad_ascent_loss, -self.max_loss, self.max_loss)
        
        return grad_ascent_loss


class UCELoss:
    """遗忘交叉熵损失函数（Unlearning Cross-Entropy Loss）。
    
    使模型对遗忘数据的预测趋向均匀分布，而不是简单地最大化损失。
    
    原理：
    - 标准 CE: loss = -sum(target * log(pred))
    - UCE: loss = -sum(target * log(1 - pred/2))
    
    当 pred 趋向均匀分布时，UCE 损失最小。
    
    参考：FedOSD 论文中的 Unlearning Cross-Entropy Loss
    
    使用方式：
    ```python
    loss_fn = UCELoss()
    loss = loss_fn(logits, labels)
    loss.backward()
    ```
    """
    
    def __init__(self, ignore_index: int = -100, reduction: str = 'mean'):
        """初始化 UCELoss。
        
        Args:
            ignore_index: 忽略的标签索引（默认 -100）
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算遗忘交叉熵损失。
        
        Args:
            logits: 模型输出 logits，形状 (batch_size, num_classes) 或 (batch_size, seq_len, vocab_size)
            labels: 目标标签，形状 (batch_size,) 或 (batch_size, seq_len)
        
        Returns:
            UCE 损失
        """
        # 展平 logits 和 labels
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        # 获取类别数
        num_classes = flat_logits.size(-1)
        
        # 处理 ignore_index
        valid_mask = flat_labels != self.ignore_index
        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]
        
        if valid_logits.numel() == 0:
            # 没有有效样本，返回零损失
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 计算 softmax 概率
        pred = F.softmax(valid_logits, dim=-1)
        
        # 创建 one-hot 编码
        target_onehot = F.one_hot(valid_labels, num_classes=num_classes).float()
        
        # 计算 UCE 损失: -sum(target * log(1 - pred/2))
        # 注意：1 - pred/2 的范围是 [0.5, 1]，确保 log 的输入为正
        loss_per_sample = -torch.sum(
            torch.log(1.0 - pred / 2 + 1e-8) * target_onehot,
            dim=-1
        )
        
        # 聚合损失
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample
