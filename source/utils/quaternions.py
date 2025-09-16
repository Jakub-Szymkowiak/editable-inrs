import torch


def quat_to_rmat(q: torch.Tensor):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    w, x, y, z = q.unbind(-1)

    N = q.size(0)
    R = torch.empty(N, 3, 3, dtype=q.dtype, device=q.device)

    # fill diag
    R[:,0,0] = 1 - 2 * (y * y + z * z)
    R[:,1,1] = 1 - 2 * (x * x + z * z)
    R[:,2,2] = 1 - 2 * (x * x + y * y)

    # fill off-diag
    R[:,0,1] = 2 * (x * y - w * z)
    R[:,1,0] = 2 * (x * y + w * z)

    R[:,0,2] = 2 * (x * z + w * y)
    R[:,2,0] = 2 * (x * z - w * y)

    R[:,1,2] = 2 * (y * z - w * x)
    R[:,2,1] = 2 * (y * z + w * x)

    return R