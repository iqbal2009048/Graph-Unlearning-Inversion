import torch
import torch.nn as nn


class KANPrivacyMask(nn.Module):
    """Concept-aware privacy masking layer using leakage scores.

    Applies a learnable mask derived from per-dimension leakage scores to
    suppress embedding dimensions that are most predictive of deleted-node
    membership.  The mask is enriched with a simplified KAN-style nonlinear
    transformation: learnable polynomial (Chebyshev) basis functions are
    applied to the leakage scores before computing the suppression mask.

    Args:
        embed_dim (int): Dimensionality of the input node embeddings.
        alpha (float): Scaling factor applied to the masked output; controls
            the trade-off between information suppression and utility
            preservation.
        poly_degree (int): Degree of the polynomial basis expansion used in
            the KAN-inspired nonlinear transformation.
    """

    def __init__(self, embed_dim: int, alpha: float = 0.5, poly_degree: int = 3):
        super(KANPrivacyMask, self).__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.poly_degree = poly_degree

        # Learnable weight per embedding dimension (used in mask computation)
        self.W = nn.Parameter(torch.ones(embed_dim))

        # KAN-style layer: learnable weights for each (dim, basis) combination
        self.kan_weights = nn.Parameter(
            torch.randn(embed_dim, poly_degree + 1) * 0.01
        )

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Chebyshev polynomials T_0 ... T_d at each element of *x*.

        Args:
            x (torch.Tensor): Input tensor of shape ``[D]`` whose values are
                assumed to lie in ``[-1, 1]`` (scores are normalized to
                ``[0, 1]`` and remapped here).

        Returns:
            torch.Tensor: Tensor of shape ``[D, poly_degree+1]``.
        """
        # Remap from [0, 1] to [-1, 1]
        x = 2.0 * x - 1.0
        polys = [torch.ones_like(x), x]
        for k in range(2, self.poly_degree + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])
        # Stack: [D, poly_degree+1]
        return torch.stack(polys[: self.poly_degree + 1], dim=-1)

    def forward(self, Z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Apply the KAN-enhanced privacy mask to the embeddings.

        Args:
            Z (torch.Tensor): Node embeddings of shape ``[N, D]``.
            S (torch.Tensor): Leakage scores of shape ``[D]``, normalized to
                ``[0, 1]`` (output of :class:`ConceptLeakageDetector`).

        Returns:
            torch.Tensor: Masked embeddings ``Z'`` of shape ``[N, D]``.
        """
        # KAN nonlinear transformation of leakage scores
        basis = self._chebyshev_basis(S)       # [D, poly_degree+1]
        S_kan = (basis * self.kan_weights).sum(dim=-1)  # [D]
        S_kan = torch.sigmoid(S_kan)            # map to (0, 1)

        # Fuse with raw leakage scores: use the mean of both signals
        S_fused = 0.5 * (S + S_kan)            # [D]

        # Learnable mask: sigmoid(W * S_fused)
        mask = torch.sigmoid(self.W * S_fused)  # [D]

        # Suppress leaky dimensions; scale by alpha
        Z_prime = Z * (1.0 - mask) * self.alpha + Z * (1.0 - self.alpha)
        return Z_prime
