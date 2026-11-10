# ResNet BasicBlock with Manifold-Constrained Hyper-Connections (mHC)
#
# Implementation of:
#   Z. Jin et al., "Beyond Residual Connections: Manifold-Constrained Hyper-Connections
#   for Robust Speaker Representation Learning", Interspeech 2026.
#
# What the paper prescribes (Sec. 2.1, 2.2, 3.2, 4.2):
#   * the hidden state is split into N parallel streams (disjoint channel groups);
#   * the identity shortcut x_{l+1} = x_l + F(x_l) is replaced by
#         h^i_{l+1} = sum_j W_ij h^j_l + f(x_l)^i ,  i = 1..N            (Eq. 1)
#     where x_l = H_pre(h_l) is the concatenation of the streams and H_post splits
#     F(x_l) back into N streams;
#   * W is a static learnable N x N matrix (not input-dependent), projected onto the
#     doubly stochastic manifold: A = exp(Theta), then alternating row / column
#     normalisation (Sinkhorn-Knopp), k = 3 iterations;
#   * N = 4 streams (best value in the ablation, Fig. 2);
#   * ResNet-34: mHC replaces the identity shortcuts between successive blocks inside
#     each stage; at stage transitions (blocks with stride / channel change) the usual
#     1x1-conv shortcut is kept ("dimension matching step").
#
# What the paper does NOT specify, and how it is resolved here:
#   * the ReLU after the addition: kept exactly as in the original BasicBlock
#     (mHC replaces only the shortcut, the block body is unchanged);
#   * the initialisation of Theta: chosen so that W starts close to the identity
#     (diagonal = MHC_INIT_DIAG), i.e. the network starts as the ordinary ResNet and
#     learns the mixing. Theta = 0 would give W = 1/N everywhere (full averaging of the
#     streams at initialisation), which removes the identity property mHC is meant to keep.


# Import of modules
import math

import torch
import torch.nn as nn
from torch.nn.functional import conv2d as F_conv2d


# Hyperparameters of mHC
MHC_NUM_STREAMS     = 4     # N, paper: N = 4 is best (Fig. 2)
MHC_SINKHORN_ITERS  = 3     # k, paper: "typically k = 3"
MHC_INIT_DIAG       = 0.95  # initial diagonal of W (not specified in the paper)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3 convolution with padding (same as in ResNetBlocks.py)

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def sinkhorn_knopp(theta, num_iters=MHC_SINKHORN_ITERS):
    # Projection of a learnable matrix onto the doubly stochastic manifold (Sec. 2.1):
    #   A = exp(Theta)
    #   repeat k times:  A <- diag(A 1)^-1 A   (row normalisation)
    #                    A <- A diag(1^T A)^-1 (column normalisation)
    # Subtracting max(Theta) does not change the result (the procedure is invariant to
    # a global scale of A) and protects exp() from overflow.

    A = torch.exp(theta - theta.max())

    for _ in range(num_iters):
        A = A/A.sum(dim=1, keepdim=True)
        A = A/A.sum(dim=0, keepdim=True)

    return A


class MHCShortcut(nn.Module):
    # Manifold-constrained hyper-connection that replaces the identity shortcut:
    #   streams h^j = channel groups of x, output stream i = sum_j W_ij h^j

    def __init__(self, channels, num_streams=MHC_NUM_STREAMS, num_iters=MHC_SINKHORN_ITERS, init_diag=MHC_INIT_DIAG):

        super(MHCShortcut, self).__init__()

        assert channels % num_streams == 0, 'Number of channels (%d) must be divisible by the number of streams (%d)'%(channels, num_streams)
        assert 1.0/num_streams < init_diag < 1.0, 'init_diag must be in (1/N, 1)'

        self.num_streams = num_streams
        self.num_iters   = num_iters

        # Theta = a*I gives exp(Theta) with e^a on the diagonal and 1 elsewhere; this matrix is
        # symmetric with equal row sums, so it is already doubly stochastic after normalisation:
        #   W_ii = e^a/(e^a + N - 1)  =>  a = log(init_diag*(N - 1)/(1 - init_diag))
        a = math.log(init_diag*(num_streams - 1)/(1.0 - init_diag))
        self.theta = nn.Parameter(a*torch.eye(num_streams))

    def mixing_matrix(self):
        # Doubly stochastic matrix W (N x N)

        return sinkhorn_knopp(self.theta, self.num_iters)

    def forward(self, x):

        C = x.size(1)
        W = self.mixing_matrix().to(x.dtype)

        # H_pre / H_post are concatenation / partition along channels: stream j is the channel group
        # [j*G, (j+1)*G), G = C/N. The mixing h_i = sum_j W_ij h_j is therefore a 1x1 convolution
        # with the weight W (x) I_G (Kronecker product):
        #     weight[i*G + a, j*G + b] = W_ij * delta_ab
        # It runs through cuDNN like the other convolutions of the network and keeps only x for
        # backward. (torch.bmm with an expanded W failed on GTX 1080 Ti with torch 1.8.1 + CUDA 11.1.)
        G = C//self.num_streams
        eye = torch.eye(G, dtype=x.dtype, device=x.device)
        weight = (W[:, None, :, None]*eye[None, :, None, :]).reshape(C, C, 1, 1)

        return F_conv2d(x, weight)

    def extra_repr(self):

        return 'num_streams={}, sinkhorn_iters={}'.format(self.num_streams, self.num_iters)


class BasicBlockMHC(nn.Module):
    # Basic ResNet block in which the identity shortcut is replaced by mHC.
    # The block body (conv3x3-BN-ReLU-conv3x3-BN, ReLU after the addition) is identical to
    # BasicBlock from ResNetBlocks.py. Blocks with a downsample shortcut (stage transitions)
    # keep the 1x1-conv shortcut, as described in the paper.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None, activation=nn.ReLU):

        super(BasicBlockMHC, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = norm_layer(planes)
        self.relu       = activation(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride

        # mHC is used only where the original block has an identity shortcut
        if self.downsample is None:
            self.mhc = MHCShortcut(planes)

        else:
            self.mhc = None

    def forward(self, x):

        if self.downsample is not None:
            identity = self.downsample(x)

        else:
            identity = self.mhc(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


def mhc_summary(model):
    # Short text summary of the learned mixing matrices (for logging)

    lines = []

    for name, module in model.named_modules():

        if isinstance(module, MHCShortcut):
            with torch.no_grad():
                W = module.mixing_matrix().cpu()
            lines.append('{:s}: mean diag {:.3f}, min diag {:.3f}, max off-diag {:.3f}'.format(name, W.diag().mean().item(), W.diag().min().item(), (W - torch.diag(W.diag())).max().item()))

    return '\n'.join(lines)
