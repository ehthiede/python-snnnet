from torch import nn
from snnnet.utils import move_from_end, move_to_end


# ------- Batch Norm Layers -------


class SnBatchNorm3d(nn.Module):
    def __init__(self, num_3d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnBatchNorm3d, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_3d_features, eps=eps, momentum=momentum,
                                         affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x


class SnBatchNorm2d(nn.Module):
    def __init__(self, num_2d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnBatchNorm2d, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_2d_features, eps=eps, momentum=momentum,
                                         affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x


class SnBatchNorm1d(nn.Module):
    def __init__(self, num_1d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnBatchNorm1d, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_1d_features, eps=eps, momentum=momentum,
                                         affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x

# ------- Instance Norm Layers -------


class SnInstanceNorm3d(nn.Module):
    def __init__(self, num_3d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnInstanceNorm3d, self).__init__()
        self.batch_norm = nn.InstanceNorm3d(num_3d_features, eps=eps, momentum=momentum,
                                            affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x


class SnInstanceNorm2d(nn.Module):
    def __init__(self, num_2d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnInstanceNorm2d, self).__init__()
        self.batch_norm = nn.InstanceNorm2d(num_2d_features, eps=eps, momentum=momentum,
                                            affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x


class SnInstanceNorm1d(nn.Module):
    def __init__(self, num_1d_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SnInstanceNorm1d, self).__init__()
        self.batch_norm = nn.InstanceNorm1d(num_1d_features, eps=eps, momentum=momentum,
                                            affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = move_from_end(x, 1)
        x = self.batch_norm(x)
        x = move_to_end(x, 1)
        return x


def _swap_axes_for_norm(a_list):
    swapped_items = [move_from_end(a, 1) for a in a_list]
    return swapped_items


def _unswap_axes_for_norm(a_list):
    swapped_items = [move_to_end(a, 1) for a in a_list]
    return swapped_items
