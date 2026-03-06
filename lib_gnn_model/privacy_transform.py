import torch
import torch.nn as nn


class MINEEstimator(nn.Module):
    """Mutual Information Neural Estimation (MINE) estimator.

    Estimates mutual information between node embeddings ``Z`` and binary
    deleted-node indicators ``Y`` using the Donsker–Varadhan lower-bound:

        I(Z; Y) ≥ E[T(Z, Y)] - log(E[exp(T(Z_shuffle, Y))])

    where ``T`` is a learned statistics network and ``Z_shuffle`` is obtained
    by shuffling ``Z`` along the batch dimension.

    Args:
        embed_dim (int): Dimensionality of the node embeddings.
        hidden_dim (int): Hidden layer size of the statistics network ``T``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super(MINEEstimator, self).__init__()
        # Statistics network T: maps (z, y) -> scalar
        self.T = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information I(Z; Y).

        Args:
            Z (torch.Tensor): Node embeddings of shape ``[N, D]``.
            Y (torch.Tensor): Binary deleted-node indicators of shape ``[N]``.

        Returns:
            torch.Tensor: Scalar MINE lower-bound estimate of I(Z; Y).
        """
        N = Z.size(0)
        Y_col = Y.float().unsqueeze(-1)   # [N, 1]

        # Joint samples (Z, Y)
        joint_input = torch.cat([Z, Y_col], dim=-1)    # [N, D+1]
        T_joint = self.T(joint_input)                  # [N, 1]

        # Marginal samples: shuffle Y independently of Z
        perm = torch.randperm(N, device=Z.device)
        Y_shuffled = Y_col[perm]
        marginal_input = torch.cat([Z, Y_shuffled], dim=-1)  # [N, D+1]
        T_marginal = self.T(marginal_input)                   # [N, 1]

        # DV lower-bound
        mi_lb = T_joint.mean() - torch.log(torch.exp(T_marginal).mean() + 1e-8)
        return mi_lb


class PrivacyCertifiedTransform(nn.Module):
    """Privacy-certified embedding transformation that minimizes MI with deleted nodes.

    Applies a lightweight linear projection to node embeddings and provides a
    regularization loss that penalizes mutual information between the
    transformed embeddings and deleted-node indicators, as estimated by
    :class:`MINEEstimator`.

    Args:
        embed_dim (int): Dimensionality of the input (and output) embeddings.
        hidden_dim (int): Hidden layer size of the internal MINE estimator.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super(PrivacyCertifiedTransform, self).__init__()
        self.embed_dim = embed_dim
        self.mine = MINEEstimator(embed_dim, hidden_dim)
        # Identity-initialized projection to keep output dimension equal
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply the certified transformation to node embeddings.

        Args:
            Z (torch.Tensor): Node embeddings of shape ``[N, D]``.

        Returns:
            torch.Tensor: Transformed embeddings of shape ``[N, D]``.
        """
        return self.proj(Z)

    def privacy_loss(self, Z: torch.Tensor, deleted_indicator: torch.Tensor) -> torch.Tensor:
        """Compute the privacy regularization loss (MINE-estimated MI).

        Args:
            Z (torch.Tensor): Transformed embeddings of shape ``[N, D]``.
            deleted_indicator (torch.Tensor): Binary tensor of shape ``[N]``
                (1 = deleted/unlearned node).

        Returns:
            torch.Tensor: Scalar privacy loss ≥ 0.
        """
        mi = self.mine(Z, deleted_indicator)
        # We want to *minimize* MI, so the loss is the MI estimate itself.
        # Clamp to avoid negative values from the lower-bound estimator.
        return torch.clamp(mi, min=0.0)
