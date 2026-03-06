import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptLeakageDetector(nn.Module):
    """Estimate which embedding dimensions leak information about deleted/unlearned nodes.

    The detector trains a small MLP classifier that predicts deleted-node membership
    from embeddings, then uses gradient-based attribution (saliency) to estimate
    feature importance per embedding dimension.

    Args:
        embed_dim (int): Dimensionality of the input node embeddings.
        hidden_dim (int): Hidden layer size for the internal MLP classifier.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super(ConceptLeakageDetector, self).__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-dimension leakage scores via gradient-based attribution.

        Args:
            Z (torch.Tensor): Node embeddings of shape ``[N, D]``.
            labels (torch.Tensor): Binary membership labels of shape ``[N]``
                (1 = deleted/unlearned node, 0 = normal node).

        Returns:
            torch.Tensor: Leakage score per dimension of shape ``[D]`` normalized
                to the range ``[0, 1]``.
        """
        Z_detached = Z.detach().requires_grad_(True)
        pred = self.classifier(Z_detached).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, labels.float())
        loss.backward()

        # Gradient-based importance: average absolute gradient over all nodes
        importance = Z_detached.grad.abs()  # [N, D]
        S = importance.mean(dim=0)          # [D]

        # Normalize to [0, 1]
        S_min = S.min()
        S_max = S.max()
        denom = S_max - S_min
        if denom > 1e-8:
            S = (S - S_min) / denom
        else:
            S = torch.zeros_like(S)

        return S.detach()

    def compute_leakage_loss(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the binary cross-entropy training loss for the internal classifier.

        This is used to train the leakage detector's weights via standard
        backpropagation.

        Args:
            Z (torch.Tensor): Node embeddings of shape ``[N, D]``.
            labels (torch.Tensor): Binary membership labels of shape ``[N]``.

        Returns:
            torch.Tensor: Scalar BCE loss.
        """
        pred = self.classifier(Z).squeeze(-1)
        return F.binary_cross_entropy_with_logits(pred, labels.float())
