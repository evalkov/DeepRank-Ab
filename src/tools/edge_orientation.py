# edge_orientation.py
#
# Utilities to build a local 3D basis from backbone atoms and express
# edge directions in that basis for residue-level graphs.

import torch
import torch.nn.functional as F


def normalize_vector(v, dim, eps=1e-6):
    """
    L2-normalize tensor v along dimension dim. Adds eps to avoid division by zero.
    """
    return v / (v.norm(p=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Project vector v onto vector e along dimension dim.
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Build an orthonormal basis from three points.
    Returns (N, 3, 3) with basis vectors [e1, e2, e3].
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    # right-handed third axis
    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)


def compute_edge_orientation(pos_CA, pos_C, pos_N, edge_index):
    """
    Compute per-edge orientation in the local node basis.

    Args:
        pos_CA: (N, 3) Cα coordinates
        pos_C:  (N, 3) C coordinates
        pos_N:  (N, 3) N coordinates
        edge_index: (2, E) long tensor with [src, dst] indices

    Returns:
        (E, 3) tensor with edge directions expressed in the source node basis.
    """
    src, dst = edge_index
    E = pos_CA[src] - pos_CA[dst]
    e_dir = F.normalize(E, p=2, dim=1)

    R_node = construct_3d_basis(pos_CA, pos_C, pos_N)
    R_edge = R_node[src]

    # batch matmul: (E, 1, 3) x (E, 3, 3) -> (E, 1, 3); then remove the singleton dim
    return torch.bmm(e_dir.unsqueeze(1), R_edge).squeeze(1)
