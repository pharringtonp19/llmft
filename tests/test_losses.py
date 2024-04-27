import torch 
import torch.nn.functional as F 
from llmft.losses import FocalLoss 

def test_focal_loss_as_cross_entropy():
    """
    Test to verify that the custom FocalLoss with gamma set to 0 behaves equivalently to
    the standard torch.nn.CrossEntropyLoss when class weights are uniform.

    This test generates a batch of logits and targets with a specified probability distribution.
    The FocalLoss is initialized with gamma=0 and uniform weights, meaning no class is favored over another.
    The test ensures that under these conditions, FocalLoss reduces to behave like CrossEntropyLoss.

    Parameters:
    - n_obs (int): Number of observations in the batch, set to 10000 for statistical stability.
    - logits (Tensor): Randomly generated logits for two classes.
    - probabilities (Tensor): A tensor of probabilities [0.7, 0.3] representing the distribution
      of two classes among the targets.
    - weights (Tensor): Uniform weights [1., 1.] used for both FocalLoss and CrossEntropyLoss
      to ensure equal treatment of all classes.
    - targets (Tensor): Targets generated based on the specified `probabilities` using multinomial distribution.

    Asserts:
    - The output of FocalLoss should be approximately equal to that of CrossEntropyLoss,
      within an absolute tolerance of 1e-3.

    The test passes if the assertion holds true, indicating functional equivalence of the two loss functions
    under the specified conditions.
    """
    n_obs = 10000 
    logits = torch.randn(n_obs, 2, requires_grad=True)
    probabilities = torch.tensor([0.7, 0.3])  
    weights =  torch.tensor([1., 1.])  
    targets = torch.multinomial(probabilities, n_obs, replacement=True)
    focal_loss_func = FocalLoss(alpha=weights, gamma=0, mode='output')
    focal_loss = focal_loss_func(logits, targets)
    ce_loss_func = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss_func(logits, targets)

    assert torch.allclose(focal_loss, ce_loss, atol=1e-3)