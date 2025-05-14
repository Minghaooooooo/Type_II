import torch
import torch.nn as nn

# set N: batch size and K : number of labels
class CustomTypeIILoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomTypeIILoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits shape: (N, K * 2) - where the last dimension represents concatenated (a, b)
        # targets shape: (N, K) - binary indicators for each class (one-hot encoded)

        # Reshape logits to get alpha (a) and beta (b)
        a = logits[:, :logits.shape[1] // 2]  # First half corresponds to a
        b = logits[:, logits.shape[1] // 2:]   # Second half corresponds to b

        type_ii_loss = (
            torch.log(a + b) -
            (targets * torch.log(a)) -
            ((1 - targets) * torch.log(b))
        ) # apply the Type II learning criterion

        loss = type_ii_loss.sum(dim=1)  # Sum over the labels for each instance

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss  #
