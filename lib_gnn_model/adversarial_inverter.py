import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Straight-through gradient reversal.

    During the **forward** pass the input is returned unchanged.  During the
    **backward** pass the gradient is multiplied by ``-lambda_`` so that the
    downstream parameters try to *maximise* whatever loss the upstream network
    is minimising.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps :class:`GradientReversalFunction` as an ``nn.Module``.

    Args:
        lambda_ (float): Scale of the gradient reversal.  A value of ``1.0``
            reverses gradients without scaling; larger values amplify the
            adversarial signal.
    """

    def __init__(self, lambda_: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialInverter(nn.Module):
    """Adversarial feature inverter trained alongside the GNN encoder.

    During training the inverter attempts to reconstruct node feature vectors
    ``X`` from (masked) node embeddings ``Z'``.  The GNN encoder receives a
    reversed gradient signal so that it learns to produce embeddings *from
    which* features cannot be reconstructed.

    The training objective from the inverter's perspective is::

        AttackLoss = MSE(X_hat, X)   # minimised by inverter

    The GNN encoder sees ``+AttackLoss`` via the gradient reversal layer,
    effectively *maximising* AttackLoss (i.e. making reconstruction harder).

    Args:
        embed_dim (int): Dimensionality of the input embeddings ``Z'``.
        feat_dim (int): Dimensionality of the node feature vectors to reconstruct.
        hidden_dim (int): Hidden layer width of the inverter MLP.
        lambda_ (float): Gradient reversal scale.
    """

    def __init__(
        self,
        embed_dim: int,
        feat_dim: int,
        hidden_dim: int = 128,
        lambda_: float = 1.0,
    ):
        super(AdversarialInverter, self).__init__()
        self.grl = GradientReversalLayer(lambda_)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, Z_prime: torch.Tensor) -> torch.Tensor:
        """Reconstruct node features from masked embeddings.

        The gradient reversal layer is applied *before* the decoder so that
        gradients flowing back to the encoder are negated.

        Args:
            Z_prime (torch.Tensor): Masked embeddings of shape ``[N, embed_dim]``.

        Returns:
            torch.Tensor: Reconstructed feature matrix ``X_hat`` of shape
                ``[N, feat_dim]``.
        """
        Z_rev = self.grl(Z_prime)
        X_hat = self.decoder(Z_rev)
        return X_hat

    def attack_loss(self, Z_prime: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute the adversarial reconstruction loss MSE(X_hat, X).

        Args:
            Z_prime (torch.Tensor): Masked embeddings of shape ``[N, embed_dim]``.
            X (torch.Tensor): Ground-truth node features of shape ``[N, feat_dim]``.

        Returns:
            torch.Tensor: Scalar MSE reconstruction loss.
        """
        X_hat = self.forward(Z_prime)
        return F.mse_loss(X_hat, X)
