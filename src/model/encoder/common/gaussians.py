import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def matrix_to_quaternion_old(
    matrix: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Convert rotation matrix (..., 3, 3) into a quaternion (..., 4),
    where the quaternion is in the order (x, y, z, w), 
    i.e., the real part is the last component.

    This matches the "SciPy format" used by your quaternion_to_matrix() function.
    """
    # matrix shape: (..., 3, 3)
    # We want output shape: (..., 4) in (x, y, z, w).

    # The trace of the matrix is the sum of diagonal elements
    batch_shape = matrix.shape[:-2]
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    trace = m00 + m11 + m22

    # Allocate output quaternions in (x, y, z, w)
    quat = matrix.new_empty((*batch_shape, 4))  # [x, y, z, w]

    # We use a "trace" approach, checking if trace > 0
    # This is a standard approach but adapted for x,y,z,w order.
    # The logic: 
    #   if (m00 + m11 + m22 + 1) > 0, 
    #   S = 2 * sqrt(1 + trace).
    #   w = 0.25 * S
    #   x = (m21 - m12) / S
    #   y = (m02 - m20) / S
    #   z = (m10 - m01) / S
    #
    # but we have to put these in the correct index for x,y,z,w.

    eps = 1e-8
    cond = (trace > 0)
    
    # ++++++++++++++
    # CASE 1: trace > 0
    # ++++++++++++++
    # S = 2 * sqrt(1 + trace)
    s_pos = 2.0 * (trace + 1.0 + eps).sqrt()
    w_pos = 0.25 * s_pos  # w
    x_pos = (matrix[..., 2, 1] - matrix[..., 1, 2]) / s_pos
    y_pos = (matrix[..., 0, 2] - matrix[..., 2, 0]) / s_pos
    z_pos = (matrix[..., 1, 0] - matrix[..., 0, 1]) / s_pos
    
    # But recall your order is (x, y, z, w) => 
    #   x = x_pos
    #   y = y_pos
    #   z = z_pos
    #   w = w_pos

    # ++++++++++++++
    # CASE 2: trace <= 0
    # ++++++++++++++
    # We find the biggest diagonal element among x, y, or z, and compute accordingly.
    # We'll do a typical approach, then reorder:

    # We'll create placeholders for the second scenario
    x_neg = torch.zeros_like(trace)
    y_neg = torch.zeros_like(trace)
    z_neg = torch.zeros_like(trace)
    w_neg = torch.zeros_like(trace)

    # Check which of m00, m11, or m22 is the largest
    cond_x = (m00 >= m11) & (m00 >= m22)  # if X is biggest
    cond_y = (~cond_x) & (m11 >= m22)    # if Y is biggest
    cond_z = (~cond_x) & (~cond_y)       # if Z is biggest

    # If x is biggest:
    #    S = sqrt(1 + m00 - m11 - m22) * 2
    #    x = 0.25 * S
    #    y = (m01 + m10) / S
    #    z = (m02 + m20) / S
    #    w = (m21 - m12) / S
    s_x = 2.0 * (1.0 + m00 - m11 - m22 + eps).sqrt()
    x_neg_x = 0.25 * s_x
    y_neg_x = (matrix[..., 0, 1] + matrix[..., 1, 0]) / s_x
    z_neg_x = (matrix[..., 0, 2] + matrix[..., 2, 0]) / s_x
    w_neg_x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / s_x

    # If y is biggest:
    #    S = sqrt(1 + m11 - m00 - m22) * 2
    #    y = 0.25 * S
    #    x = (m01 + m10) / S
    #    z = (m12 + m21) / S
    #    w = (m02 - m20) / S
    s_y = 2.0 * (1.0 + m11 - m00 - m22 + eps).sqrt()
    y_neg_y = 0.25 * s_y
    x_neg_y = (matrix[..., 0, 1] + matrix[..., 1, 0]) / s_y
    z_neg_y = (matrix[..., 1, 2] + matrix[..., 2, 1]) / s_y
    w_neg_y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / s_y

    # If z is biggest:
    #    S = sqrt(1 + m22 - m00 - m11) * 2
    #    z = 0.25 * S
    #    x = (m02 + m20) / S
    #    y = (m12 + m21) / S
    #    w = (m10 - m01) / S
    s_z = 2.0 * (1.0 + m22 - m00 - m11 + eps).sqrt()
    z_neg_z = 0.25 * s_z
    x_neg_z = (matrix[..., 0, 2] + matrix[..., 2, 0]) / s_z
    y_neg_z = (matrix[..., 1, 2] + matrix[..., 2, 1]) / s_z
    w_neg_z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / s_z

    # Merge them based on cond_x, cond_y, cond_z
    x_neg = torch.where(cond_x, x_neg_x, x_neg)
    y_neg = torch.where(cond_x, y_neg_x, y_neg)
    z_neg = torch.where(cond_x, z_neg_x, z_neg)
    w_neg = torch.where(cond_x, w_neg_x, w_neg)

    x_neg = torch.where(cond_y, x_neg_y, x_neg)
    y_neg = torch.where(cond_y, y_neg_y, y_neg)
    z_neg = torch.where(cond_y, z_neg_y, z_neg)
    w_neg = torch.where(cond_y, w_neg_y, w_neg)

    x_neg = torch.where(cond_z, x_neg_z, x_neg)
    y_neg = torch.where(cond_z, y_neg_z, y_neg)
    z_neg = torch.where(cond_z, z_neg_z, z_neg)
    w_neg = torch.where(cond_z, w_neg_z, w_neg)

    # Now combine the trace>0 (pos) vs. trace<=0 (neg)
    x_val = torch.where(cond, x_pos, x_neg)
    y_val = torch.where(cond, y_pos, y_neg)
    z_val = torch.where(cond, z_pos, z_neg)
    w_val = torch.where(cond, w_pos, w_neg)

    # Put them in the correct order: (x, y, z, w)
    quat = torch.stack([x_val, y_val, z_val, w_val], dim=-1)

    # Optionally normalize to reduce numeric drift
    quat = quat / (quat.norm(dim=-1, keepdim=True) + eps)
    return quat

def matrix_to_quaternion(R):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    Input shape: (..., 3, 3)
    Output shape: (..., 4) (w, x, y, z)
    """
    batch_dim = R.shape[:-2]  # Preserve batch dimensions
    R_flat = R.reshape(-1, 3, 3)  # Flatten batch dimensions for easier computation

    trace = R_flat[..., 0, 0] + R_flat[..., 1, 1] + R_flat[..., 2, 2]
    quaternions = torch.zeros((*R_flat.shape[:-2], 4), device=R.device)

    # Case 1: trace > 0
    mask = trace > 0
    S = torch.sqrt(1.0 + trace[mask]) * 2  # 4w
    quaternions[mask, 0] = 0.25 * S
    quaternions[mask, 1] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) / S
    quaternions[mask, 2] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) / S
    quaternions[mask, 3] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) / S

    # Case 2: R[0,0] is the largest diagonal term
    mask = (~mask) & (R_flat[..., 0, 0] > R_flat[..., 1, 1]) & (R_flat[..., 0, 0] > R_flat[..., 2, 2])
    S = torch.sqrt(1.0 + R_flat[mask, 0, 0] - R_flat[mask, 1, 1] - R_flat[mask, 2, 2]) * 2  # 4x
    quaternions[mask, 0] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) / S
    quaternions[mask, 1] = 0.25 * S
    quaternions[mask, 2] = (R_flat[mask, 0, 1] + R_flat[mask, 1, 0]) / S
    quaternions[mask, 3] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) / S

    # Case 3: R[1,1] is the largest diagonal term
    mask = (~mask) & (R_flat[..., 1, 1] > R_flat[..., 2, 2])
    S = torch.sqrt(1.0 + R_flat[mask, 1, 1] - R_flat[mask, 0, 0] - R_flat[mask, 2, 2]) * 2  # 4y
    quaternions[mask, 0] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) / S
    quaternions[mask, 1] = (R_flat[mask, 0, 1] + R_flat[mask, 1, 0]) / S
    quaternions[mask, 2] = 0.25 * S
    quaternions[mask, 3] = (R_flat[mask, 1, 2] + R_flat[mask, 2, 1]) / S

    # Case 4: R[2,2] is the largest diagonal term
    mask = ~mask
    S = torch.sqrt(1.0 + R_flat[mask, 2, 2] - R_flat[mask, 0, 0] - R_flat[mask, 1, 1]) * 2  # 4z
    quaternions[mask, 0] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) / S
    quaternions[mask, 1] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) / S
    quaternions[mask, 2] = (R_flat[mask, 1, 2] + R_flat[mask, 2, 1]) / S
    quaternions[mask, 3] = 0.25 * S

    return quaternions.reshape(*batch_dim, 4)  # Restore batch dimensions

