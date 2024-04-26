import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', mode='output'):
        """
        Args:
            alpha (Tensor, optional): Weights tensor. If mode is 'output', weights are applied based on target labels.
                                      If mode is 'input', weights are based on type of inputs.
            gamma (float): Focusing parameter.
            reduction (str): Method for reducing the loss to a single value ('none', 'mean', or 'sum').
            mode (str): 'output' for applying weights based on the targets, 'input' for applying weights based on input types.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mode = mode

    def forward(self, inputs, targets, type_indicator=None):
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the predicted probabilities
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1))

        # Calculate focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha weighting
        if self.alpha is not None:
            if self.mode == 'input':
                if type_indicator is None:
                    raise ValueError("type_indicator is required when mode is 'input'")
                at = self.alpha[type_indicator]
            else:  # default to 'output'
                at = self.alpha[targets]

            focal_loss *= at

        # Aggregate the loss based on the reduction method
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
