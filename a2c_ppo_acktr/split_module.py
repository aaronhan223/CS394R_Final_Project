import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import *
from a2c_ppo_acktr.utils import init

###############################################################################
#
# Split Network
#
###############################################################################

class SplitNet(nn.Module):
    def __init__(self):
        super(SplitNet, self).__init__()
        self.net = None
        self.n_elites = 1
        self.next_layers = {}
        self.layers_to_split = []

    def create_optimizer(self):
        pass

    def forward(self, x, split_layer=-1):
        pass

    def loss_fn(self, split_layer=-1):
        pass

    def update(self, x, y):
        pass

    def split(self):
        pass

    def sp_where(self):
        w_list = [self.net[layer].w for layer in self.layers_to_split]
        threshold = np.sort(np.concatenate(w_list).reshape(-1))[self.n_elites]
        return threshold

    def save(self, path='./tmp.pt'):
        torch.save(self.state_dict(), path)

    def load(self, path='./tmp.pt'):
        self.load_state_dict(torch.load(path))

    def get_num_params(self):
        model_n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return model_n_params

###############################################################################
#
# Split Block Abstract Class
#
###############################################################################

class SplitModule(nn.Module):
    def __init__(self, can_split=True, actv_fn='relu', has_bn=False):
        super(SplitModule, self).__init__()
        self.can_split = can_split
        self.actv_fn = actv_fn
        self.has_bn = has_bn
        self.bn = None
        self.w = None
        self.v = None
        self.Ys = []
        self.module = None

    def _reset_Ys(self):
        for y in self.Ys:
            del y
        self.Ys = []

    def _d2_actv(self, x, beta=3.):
        if self.actv_fn == 'relu':
            # use 2nd order derivative of softplus for approximation
            s = torch.sigmoid(x*beta)
            return beta*s*(1.-s)
        elif self.actv_fn == 'swish':
            s = torch.sigmoid(x)
            return s*(1.-s) + s + x*s*(1.-s) - (s.pow(2) + 2.*x*s.pow(2)*(1.-s))
        elif self.actv_fn == 'sigmoid':
            s = torch.sigmoid(x)
            return (s-s.pow(2)) * (1.-s).pow(2)
        elif self.actv_fn == 'tanh':
            h = torch.tanh(x)
            return -2.*h * (1-h.pow(2))
        elif self.actv_fn == 'none':
            return torch.ones_like(x)
        else:
            raise Exception('[ERROR] unknown activation')

    def _activate(self, x):
        if self.actv_fn == 'relu':
            return F.relu(x)
        elif self.actv_fn == 'swish':
            return x * torch.sigmoid(x)
        elif self.actv_fn == 'sigmoid':
            return torch.sigmoid(x)
        elif self.actv_fn == 'tanh':
            return torch.tanh(x)
        elif self.actv_fn == 'none':
            return x
        else:
            raise Exception('[ERROR] unknown activation')

    def sp_eigen(self, avg_over=1.):
        pass

    def forward(self, x, pre_split=False):
        pass

    def sp_forward(self, x):
        """
        the forward pass that determines the splitting matrix.
        """
        if self.has_bn:
            self.bn.eval()
        pass

    def active_split(self, threshold):
        """
        actively split the current layer.
        @param threshold: neurons w/ eigenvalues above threshold should split.
        """
        pass

    def passive_split(self, idx):
        """
        passively split due to the splitting of a previous layer.
        @param idx: which dimensions should be duplicated.
        """
        pass

###############################################################################
#
# Different Split Layers
#
###############################################################################

class SpLinear(SplitModule):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_type=3,
                 can_split=True,
                 actv_fn='relu',
                 has_bn=False):

        super().__init__(can_split=can_split, actv_fn=actv_fn, has_bn=has_bn)

        self.has_bias = bias
        if has_bn:
            self.bn = nn.BatchNorm1d(out_features)
            self.has_bias = False

        init1_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        init2_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        init3_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.module = nn.Linear(in_features, out_features, self.has_bias)
        init_dict = {
            1: init1_,
            2: init2_,
            3: init3_,
        }
        init_dict[init_type](self.module)

    def forward(self, x, pre_split=False):
        if pre_split and self.can_split:
            return self.sp_forward(x)
        x = self.module(x)
        if self.has_bn:
            x = self.bn(x)
        x = self._activate(x)
        return x

    def sp_forward(self, x):
        """
        Notice that for a single neuron:
            tr(Y nabla2_(sigma(x))) = tr(Y nabla2(sigma) xx^T) = nabla2(sigma) xYx^T
            nabla2 denote the 2nd order derivative
        x: [B, H_in]
        """
        out = self.module(x) # [B, H_out]
        if self.has_bn:
            self.bn.eval() # fix running mean/variance
            out = self.bn(out)
            # calculate bn_coff
            bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight
            bn_coff = bn_coff.view(1, -1) # [1, n_out]

        first_run = (len(self.Ys) == 0)

        # calculate 2nd order derivative of the activation
        nabla2_out = self._d2_actv(out) # [B, H_out]
        B, H_in = x.shape; H_out = out.shape[1]

        auxs = [] # separate calculations for each neuron for space efficiency
        for neuron_idx in range(H_out):
            c = bn_coff[:, neuron_idx:neuron_idx + 1] if self.has_bn else 1.
            if first_run:
                Y = Variable(torch.zeros(H_in, H_in).cuda(), requires_grad=True) # [H_in, H_in]
                self.Ys.append(Y)
            else:
                Y = self.Ys[neuron_idx]
            try:
                aux = c * x.mm(Y).unsqueeze(1).bmm(c * x.unsqueeze(-1)).squeeze(-1) # (Bx)Y(Bx^T), [B, 1]
            except:
                import pdb; pdb.set_trace()
            auxs.append(aux)

        auxs = torch.cat(auxs, -1) # [B, H_out]
        auxs = auxs * nabla2_out # [B, H_out]
        out = self._activate(out) + auxs
        return out

    def sp_eigen(self, avg_over=1.):
        #import pdb; pdb.set_trace()
        A = np.array([item.grad.data.cpu().numpy() for item in self.Ys]) # [n_neurons, H_in, H_in]
        A /= avg_over
        A = (A + np.transpose(A, [0, 2, 1])) / 2
        w, v = np.linalg.eig(A) # [H_out, K], [H_out, H_in, K]
        w = np.real(w)
        v = np.real(v)
        min_idx = np.argmin(w, axis=1)
        w_min = np.min(w, axis=1) # [H_out,]
        v_min = v[np.arange(w_min.shape[0]), :, min_idx] # [H_out, H_in]
        self.w = w_min
        self.v = v_min
        self._reset_Ys()

    def random_split(self, H_new):
        if H_new == 0:
            return 0, None

        H_out, H_in = self.module.weight.shape
        idx = np.random.choice(H_out, H_new)
        device = 'cuda' if self.module.weight.is_cuda else 'cpu'

        delta = torch.randn(H_new, H_in).to(device) * 0.1
        idx = torch.LongTensor(idx).to(device)

        new_layer = nn.Linear(H_in, H_out+H_new, bias=self.has_bias).to(device)

        # for current layer
        new_layer.weight.data[:H_out, :] = self.module.weight.data.clone()
        new_layer.weight.data[H_out:, :] = self.module.weight.data[idx, :]
        new_layer.weight.data[idx, :] += 1e-2 * delta
        new_layer.weight.data[H_out:, :] -= 1e-2 * delta

        if self.has_bias:
            new_layer.bias.data[:H_out] = self.module.bias.data.clone()
            new_layer.bias.data[H_out:] = self.module.bias.data[idx]

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm1d(H_out+H_new).to(device)
            new_bn.weight.data[:H_out] = self.bn.weight.data.clone()
            new_bn.weight.data[H_out:] = self.bn.weight.data[idx]
            new_bn.bias.data[:H_out] = self.bn.bias.data.clone()
            new_bn.bias.data[H_out:] = self.bn.bias.data[idx]
            new_bn.running_mean.data[:H_out] = self.bn.running_mean.data.clone()
            new_bn.running_mean.data[H_out:] = self.bn.running_mean.data[idx]
            new_bn.running_var.data[:H_out] = self.bn.running_var.data.clone()
            new_bn.running_var.data[H_out:] = self.bn.running_var.data[idx]
            self.bn = new_bn
        return H_new, idx

    def active_split(self, threshold):
        idx = np.argwhere(self.w < threshold).reshape(-1) # those are neurons ready for splitting
        H_new = len(idx)
        if H_new == 0:
            return 0, None

        H_out, H_in = self.module.weight.shape
        device = 'cuda' if self.module.weight.is_cuda else 'cpu'

        delta = torch.Tensor(self.v[idx, :]).to(device)
        idx = torch.LongTensor(idx).to(device)

        new_layer = nn.Linear(H_in, H_out+H_new, bias=self.has_bias).to(device)

        # for current layer
        new_layer.weight.data[:H_out, :] = self.module.weight.data.clone()
        new_layer.weight.data[H_out:, :] = self.module.weight.data[idx, :]
        new_layer.weight.data[idx, :] += 1e-1 * delta
        new_layer.weight.data[H_out:, :] -= 1e-1 * delta

        if self.has_bias:
            new_layer.bias.data[:H_out] = self.module.bias.data.clone()
            new_layer.bias.data[H_out:] = self.module.bias.data[idx]

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm1d(H_out+H_new).to(device)
            new_bn.weight.data[:H_out] = self.bn.weight.data.clone()
            new_bn.weight.data[H_out:] = self.bn.weight.data[idx]
            new_bn.bias.data[:H_out] = self.bn.bias.data.clone()
            new_bn.bias.data[H_out:] = self.bn.bias.data[idx]
            new_bn.running_mean.data[:H_out] = self.bn.running_mean.data.clone()
            new_bn.running_mean.data[H_out:] = self.bn.running_mean.data[idx]
            new_bn.running_var.data[:H_out] = self.bn.running_var.data.clone()
            new_bn.running_var.data[H_out:] = self.bn.running_var.data[idx]
            self.bn = new_bn
        return H_new, idx

    def passive_split(self, idx):
        H_new = idx.shape[0]
        H_out, H_in = self.module.weight.shape
        device = 'cuda' if self.module.weight.is_cuda else 'cpu'
        new_layer = nn.Linear(H_in+H_new, H_out, bias=self.has_bias).to(device)
        new_layer.weight.data[:, :H_in] = self.module.weight.data.clone()
        new_layer.weight.data[:, H_in:] = self.module.weight.data[:, idx] / 2.
        new_layer.weight.data[:, idx] /= 2.
        if self.has_bias:
            new_layer.bias.data = self.module.bias.data.clone()
        self.module = new_layer


class SpConv2d(SplitModule):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            can_split=True,
            bias=True,
            actv_fn='relu',
            has_bn=False):

        super().__init__(can_split=can_split, actv_fn=actv_fn, has_bn=has_bn)

        self.has_bias = bias
        if has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self.has_bias = False

        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kh, self.kw = kernel_size

        if isinstance(padding, int):
            self.ph = self.pw = padding
        else:
            assert len(padding) == 2
            self.ph, self.pw = padding

        if isinstance(stride, int):
            self.dh = self.dw = stride
        else:
            assert len(stride) == 2
            self.dh, self.dw = stride

        self.module = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=self.has_bias)

    def forward(self, x, pre_split=False):
        if pre_split and self.can_split:
            return self.sp_forward(x)

        x = self.module(x)
        if self.has_bn:
            x = self.bn(x)
        x = self._activate(x)
        return x

    def get_conv_patches(self, x):
        # Pad tensor to get the same output
        x = F.pad(input, (self.pw, self.pw, self.ph, self.ph))  #(padding_left, padding_right, padding_top, padding_bottom)
        # get all image windows of size (kh, kw) and stride (dh, dw)
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        # Permute so that channels are next to patch dimension
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [batch_size, h, w, n_in, kh, kw]
        return patches

    def sp_forward(self, x):
        """
        Notice that for a single neuron:
            tr(Y nabla2_(sigma(x))) = tr(Y nabla2(sigma) xx^T) = nabla2(sigma) xYx^T
            nabla2 denote the 2nd order derivative
        x: [B, C_in, H, W]
        """
        out = self.module(x) # [B, C_out, H, W]

        if self.has_bn:
            self.bn.eval() # fix running mean/variance
            out = self.bn(out)
            # calculate bn_coff
            bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight
            bn_coff = bn_coff.view(1, -1, 1, 1) # [1, C_out, 1, 1]

        first_run = (len(self.Ys) == 0)

        # calculate 2nd order derivative of the activation
        nabla2_out = self._d2_actv(out) # [B, C_out, H, W]
        patches = self.get_conv_patches(x)
        B, H, W, C_in, KH, KW = patches.size()
        C_out = out.shape[1]

        D = C_in * KH * KW
        x = patches.view(B, H, W, D)

        auxs = [] # separate calculations for each neuron for space efficiency
        for neuron_idx in range(C_out):
            c = bn_coff[:, neuron_idx:neuron_idx+1, :, :] if self.has_bn else 1.
            l = c * x
            if first_run:
                Y = Variable(torch.zeros(D, D).cuda(), requires_grad=True) # [H_in, H_in]
                self.Ys.append(Y)
            else:
                Y = self.Ys[neuron_idx]
            aux = l.view(-1, D).mm(Y).unsqueeze(1).bmm(l.view(-1, D, 1)).squeeze(-1) # (Bx)Y(Bx^T), [B*H*W,1]
            aux = aux.view(B, 1, H, W)
            auxs.append(aux)

        auxs = torch.cat(auxs, 1) # [B, C_out, H, W]
        auxs = auxs * nabla2_out # [B, C_out, H, W]
        out = self._activate(out) + auxs
        return out

    def sp_eigen(self, avg_over=1.):
        A = np.array([item.grad.data.cpu().numpy() for item in self.Ys]) # [C_out, D, D]
        A /= avg_over
        A = (A + np.transpose(A, [0, 2, 1])) / 2
        w, v = np.linalg.eig(A) # [C_out, K], [C_out, D, K]
        w = np.real(w)
        v = np.real(v)
        min_idx = np.argmin(w, axis=1)
        w_min = np.min(w, axis=1) # [C_out,]
        v_min = v[np.arange(w_min.shape[0]), :, min_idx] # [C_out, D]
        self.w = w_min
        self.v = v_min
        self._reset_Ys()

    def active_split(self, threshold):
        idx = np.argwhere(self.w < threshold).reshape(-1) # those are neurons ready for splitting
        C_new = len(idx)
        if C_new == 0:
            return 0, None

        C_out, C_in, _, _ = self.module.weight.shape
        device = 'cuda' if self.module.weight.is_cuda else 'cpu'

        delta = torch.Tensor(self.v[idx, ...]).to(device).view(-1, C_in, self.kh, self.kw)
        idx = torch.LongTensor(idx).to(device)

        new_layer = nn.Conv2d(in_channels=C_in,
                              out_channels=C_out+C_new,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, dw),
                              padding=(self.ph, pw),
                              bias=self.has_bias).to(device)

        # for current layer
        new_layer.weight.data[:C_out, ...] = self.module.weight.data.clone()
        new_layer.weight.data[C_out:, ...] = self.module.weight.data[idx, ...]
        new_layer.weight.data[idx, ...] += 1e-2 * delta
        new_layer.weight.data[C_out:, ...] -= 1e-2 * delta

        if self.has_bias:
            new_layer.bias.data[:C_out, ...] = self.module.bias.data.clone()
            new_layer.bias.data[C_out:, ...] = self.module.bias.data[idx]

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm2d(C_out+C_new).to(device)
            new_bn.weight.data[:C_out] = self.bn.weight.data.clone()
            new_bn.weight.data[C_out:] = self.bn.weight.data[idx]
            new_bn.bias.data[:C_out] = self.bn.bias.data.clone()
            new_bn.bias.data[C_out:] = self.bn.bias.data[idx]
            new_bn.running_mean.data[:C_out] = self.bn.running_mean.data.clone()
            new_bn.running_mean.data[C_out:] = self.bn.running_mean.data[idx]
            new_bn.running_var.data[:C_out] = self.bn.running_var.data.clone()
            new_bn.running_var.data[C_out:] = self.bn.running_var.data[idx]
            self.bn = new_bn
        return H_new, idx

    def passive_split(self, idx):
        C_new = idx.shape[0]
        C_out, C_in, _, _ = self.module.weight.shape
        device = 'cuda' if self.module.weight.is_cuda else 'cpu'
        new_layer = nn.Conv2d(in_channels=C_in+C_new,
                              out_channels=C_out,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, dw),
                              padding=(self.ph, pw),
                              bias=self.has_bias).to(device)

        new_layer.weight.data[:, :H_in, ...] = self.module.weight.data.clone()
        new_layer.weight.data[:, H_in:, ...] = self.module.weight.data[:, idx, ...] / 2.
        new_layer.weight.data[:, idx, ...] /= 2.
        if self.has_bias:
            new_layer.bias.data = self.module.bias.data.clone()
        self.module = new_layer
