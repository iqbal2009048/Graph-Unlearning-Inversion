import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
import numpy as np

from typing import NamedTuple, List, Tuple


def _metis_available() -> bool:
    """Return True if torch_sparse METIS partitioning is usable."""
    try:
        import torch_sparse  # noqa: F401
        partition_fn = torch.ops.torch_sparse.partition
        # Do a minimal test call to check METIS support at runtime
        rowptr = torch.tensor([0, 1, 2], dtype=torch.long)
        col = torch.tensor([1, 0], dtype=torch.long)
        partition_fn(rowptr, col, None, 2, False)
        return True
    except Exception:
        return False


class DatasetPartitioner:
    def __init__(self, data: Data, args):
        self.data = data
        self.partition_method = args['partition_method']
        self.split_ratio = args['random_part_ratio']
        self.metis_parts = args['metis_parts']
        self.metis_shadow_parts = args['metis_shadow_parts']

    def split(self) -> Tuple[Data, Data]:
        if self.partition_method == 'random':
            return self.random_split()
        elif self.partition_method == 'metis':
            return self.metis_split()

    def random_split(self) -> Tuple[Data, Data]:
        """Randomly splits the dataset into shadow and attack datasets."""
        num_nodes = self.data.num_nodes
        indices = torch.randperm(num_nodes)
        split_idx = int(self.split_ratio * num_nodes)

        shadow_nodes = indices[:split_idx]
        attack_nodes = indices[split_idx:]

        return *self._create_subgraphs(shadow_nodes, attack_nodes), shadow_nodes, attack_nodes

    def label_propagation_split(self) -> Tuple[Data, Data]:
        raise NotImplementedError('LPSplit: To be implemented')

    def metis_split(self, recursive: bool = False) -> Tuple[Data, Data]:
        if self.metis_shadow_parts >= self.metis_parts:
            raise ValueError("Number of shadow parts must be less than the total number of parts.")

        if _metis_available():
            perm, ptr = self._metis_partition_torch_sparse(
                self.data.edge_index, self.data.num_nodes,
                num_parts=self.metis_parts, recursive=recursive
            )
        else:
            # Fallback: use torch_geometric ClusterData if pyg-lib is available,
            # otherwise fall back to a random split with a warning.
            try:
                from torch_geometric.loader import ClusterData
                perm, ptr = self._metis_partition_cluster_data(
                    num_parts=self.metis_parts, recursive=recursive
                )
            except (ImportError, RuntimeError):
                import warnings
                warnings.warn(
                    "METIS partitioning requires 'torch-sparse' or 'pyg-lib'. "
                    "Falling back to random partitioning.",
                    RuntimeWarning,
                )
                return self._random_split_n_parts(
                    self.metis_parts, self.metis_shadow_parts
                )

        # Assign shadow and attack nodes based on partition clusters
        shadow_nodes = perm[ptr[:self.metis_shadow_parts].sum():ptr[self.metis_shadow_parts].sum()]
        attack_nodes = perm[ptr[self.metis_shadow_parts].sum():ptr[self.metis_parts].sum()]

        return *self._create_subgraphs(shadow_nodes, attack_nodes), shadow_nodes, attack_nodes

    def _metis_partition_torch_sparse(
        self, edge_index: torch.Tensor, num_nodes: int,
        num_parts: int, recursive: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """METIS partitioning via torch_sparse ops."""
        from torch_sparse import SparseTensor
        adj_t = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(num_nodes, num_nodes)
        )
        rowptr, col, _ = adj_t.csr()
        cluster = torch.ops.torch_sparse.partition(rowptr, col, None, num_parts, recursive)
        cluster, perm = cluster.sort()
        ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)
        return perm, ptr

    def _metis_partition_cluster_data(
        self, num_parts: int, recursive: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """METIS partitioning via torch_geometric ClusterData."""
        from torch_geometric.loader import ClusterData
        cluster_data = ClusterData(
            self.data, num_parts=num_parts, recursive=recursive, log=False
        )
        perm = cluster_data.partition.node_perm
        ptr = cluster_data.partition.partptr
        return perm, ptr

    def _random_split_n_parts(
        self, num_parts: int, shadow_parts: int
    ) -> Tuple[Data, Data, torch.Tensor, torch.Tensor]:
        """Random fallback that mimics an N-way split.

        Ensures at least one node ends up in each partition.
        """
        num_nodes = self.data.num_nodes
        indices = torch.randperm(num_nodes)
        # Clamp so that both shadow and attack partitions are non-empty
        shadow_end = max(1, min(int((shadow_parts / num_parts) * num_nodes), num_nodes - 1))
        shadow_nodes = indices[:shadow_end]
        attack_nodes = indices[shadow_end:]
        return *self._create_subgraphs(shadow_nodes, attack_nodes), shadow_nodes, attack_nodes

    def _create_subgraphs(self, shadow_nodes: torch.Tensor, attack_nodes: torch.Tensor) -> Tuple[Data, Data]:
        """Creates shadow and attack subgraphs without overlapping nodes or edges."""
        shadow_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        shadow_mask[shadow_nodes] = True

        attack_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        attack_mask[attack_nodes] = True

        shadow_data = Data(
            x=self.data.x[shadow_mask],
            edge_index=self._filter_edges(self.data.edge_index, shadow_mask),
            y=self.data.y[shadow_mask]
        )

        attack_data = Data(
            x=self.data.x[attack_mask],
            edge_index=self._filter_edges(self.data.edge_index, attack_mask),
            y=self.data.y[attack_mask]
        )

        return shadow_data, attack_data

    def _filter_edges(self, edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Filters and reindexes edges based on the node mask."""
        # Map original node indices to new indices
        new_index = torch.full((node_mask.size(0),), -1, dtype=torch.long)
        new_index[node_mask] = torch.arange(node_mask.sum())

        # Filter edges
        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        filtered_edge_index = edge_index[:, mask]
        
        # Reindex edges
        reindexed_edge_index = new_index[filtered_edge_index]

        return reindexed_edge_index


# A test case for the DatasetPartitioner
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch_geometric.datasets import Planetoid, KarateClub
    from torch_geometric.utils import to_networkx
    import networkx as nx

    x = torch.arange(7).unsqueeze(-1)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6],
        [1, 2, 2, 0, 6, 1, 0, 4, 5, 6, 3, 5, 3, 4, 3, 4, 1]
    ])
    data = Data(x=x, edge_index=edge_index)

    # Instantiate the partitioner
    partitioner = DatasetPartitioner(data, seed=7)

    # Test random split
    shadow_data, attack_data = partitioner.metis_split(2, 1)

    # Convert subgraphs to NetworkX for visualization
    G = to_networkx(data, to_undirected=True)
    G_shadow = to_networkx(shadow_data, to_undirected=True)
    G_attack = to_networkx(attack_data, to_undirected=True)

    # Plot the splits
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42)  # Layout for consistent node positioning

    # Original graph
    plt.subplot(1, 3, 1)
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray')
    plt.title("Original Graph")

    # Shadow graph
    plt.subplot(1, 3, 2)
    nx.draw(G_shadow, pos, with_labels=True, node_color='blue', edge_color='lightblue')
    plt.title("Shadow Graph")

    # Attack graph
    plt.subplot(1, 3, 3)
    nx.draw(G_attack, pos, with_labels=True, node_color='red', edge_color='pink')
    plt.title("Attack Graph")

    plt.tight_layout()
    plt.show()
