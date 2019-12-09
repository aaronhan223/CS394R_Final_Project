import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import init, AddBias
from a2c_ppo_acktr.split_module import *

import math
import time


"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(SplitNet):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, dimh=2):
        super(Policy, self).__init__()

        net = None
        
        dim_input = obs_shape[0]
        if len(obs_shape) == 1:
            net = [
                # actor base
                SpLinear(dim_input, dimh, actv_fn='tanh'),
                SpLinear(dimh, dimh, actv_fn='tanh'),
                # critic base
                SpLinear(dim_input, dimh, actv_fn='tanh'),
                SpLinear(dimh, dimh, actv_fn='tanh'),
                SpLinear(dimh, 1, actv_fn='none'),
            ]
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            self.action_type = 'discrete'
            num_outputs = action_space.n
            net.append(
                SpLinear(dimh, num_outputs, actv_fn='none', init_type=1)
            )
        elif action_space.__class__.__name__ == "Box":
            self.action_type = 'continuous'
            num_outputs = action_space.shape[0]
            net.append(
                SpLinear(dimh, num_outputs, actv_fn='none', init_type=2)
            )
            self.logstd = AddBias(torch.zeros(num_outputs))
        else:
            raise NotImplementedError

        self.net = nn.ModuleList(net)
        self.next_layer = {
            0: [1],
            1: [5],
            2: [3],
            3: [4],
        }
        '''
        self.next_layer = {
            0: [3],
            1: [2],
        }
        '''
        self.layers_to_split = list(self.next_layer.keys())
        self.n_elites = 64
        #self.eigen_threshold =

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def base_output(self, inputs, rnn_hxs, masks, split=False):
        #print('in base ouptu, split is ', split)
        x = inputs
        hidden_critic = self.net[3](self.net[2](x, split), split)
        hidden_actor = self.net[1](self.net[0](x, split), split)
        critic_out = self.net[4](hidden_critic)

        #critic_out = self.net[2](self.net[1](x, split), split)
        #hidden_actor = self.net[0](x, split)
        return critic_out, hidden_actor, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base_output(inputs, rnn_hxs, masks)
        x = self.net[-1](actor_features)
        dist = None
        if self.action_type == 'discrete':
            dist = FixedCategorical(logits=x)
        elif self.action_type == 'continuous':
            zeros = torch.zeros(x.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
            dist = FixedNormal(x, action_logstd.exp())

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, split=False):
        value, _, _ = self.base_output(inputs, rnn_hxs, masks, split)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, split=False):
        #print('in eval action split is ', split)
        value, actor_features, rnn_hxs = self.base_output(inputs, rnn_hxs, masks, split)
        x = self.net[-1](actor_features)

        dist = None
        if self.action_type == 'discrete':
            dist = FixedCategorical(logits=x)
        elif self.action_type == 'continuous':
            zeros = torch.zeros(x.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
            dist = FixedNormal(x, action_logstd.exp())

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs

    def get_num_params(self):
        #model_n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #return model_n_params
        a1 = self.net[0].module.weight.shape[0]
        a2 = self.net[1].module.weight.shape[0]
        c1 = self.net[2].module.weight.shape[0]
        c2 = self.net[3].module.weight.shape[0]
        #return [a1, a2, c1, c2]
        a = a1 + a2
        c = c1 + c2
        return [a, c]

    def random_split(self):
        t0 = time.time()

        n_neurons_added = {}
        n_layers_to_split = len(self.layers_to_split)
        n_news = np.random.multinomial(self.n_elites, [1/float(n_layers_to_split)]*n_layers_to_split)
        # -- actual splitting -- #
        for i in reversed(self.layers_to_split): # i is the i-th module in self.net
            n_new, idx = self.net[i].random_split(n_news[i])
            n_neurons_added[i] = n_new
            if n_new > 0: # we have indeed splitted this layer
                for j in self.next_layer[i]:
                    self.net[j].passive_split(idx)

        for net in self.net:
            net._reset_Ys()

        t1 = time.time()
        '''
        print('[INFO] random splitting takes %10.4f sec.' % (t1-t0))
        print('[INFO] number of added neurons:')
        print(' -------- actor --------')
        print(' -- layer %d grows %d neurons' % (1, n_neurons_added[0]))
        print(' -- layer %d grows %d neurons' % (2, n_neurons_added[1]))
        print(' -- actor output layer pi (|A|) --')
        print(' -------- critic -------')
        print(' -- layer %d grows %d neurons' % (1, n_neurons_added[2]))
        print(' -- layer %d grows %d neurons' % (2, n_neurons_added[3]))
        print(' -- critic output layer V (1) --')
        '''

    def split(self, avg_over=1.):
        t0 = time.time()
        for i in self.layers_to_split: # for each layer to split, go over the data to determine the eigenvalues
            self.net[i].sp_eigen(avg_over)

        # -- calculate the cutoff threshold for determining whom to split -- #
        threshold = self.sp_where()

        n_neurons_added = {}
        # -- actual splitting -- #
        for i in reversed(self.layers_to_split): # i is the i-th module in self.net
            n_new, idx = self.net[i].active_split(threshold)
            n_neurons_added[i] = n_new
            if n_new > 0: # we have indeed splitted this layer
                for j in self.next_layer[i]:
                    self.net[j].passive_split(idx)

        for net in self.net:
            net._reset_Ys()

        t1 = time.time()
        
        print('[INFO] splitting takes %10.4f sec. Threshold eigenvalue is %10.4f' % (t1-t0, threshold))
        print('[INFO] number of added neurons:')
        print(' -------- actor --------')
        print(' -- layer %d grows %d neurons' % (1, n_neurons_added[0]))
        print(' -- layer %d grows %d neurons' % (2, n_neurons_added[1]))
        print(' -- actor output layer pi (|A|) --')
        print(' -------- critic -------')
        print(' -- layer %d grows %d neurons' % (1, n_neurons_added[2]))
        print(' -- layer %d grows %d neurons' % (2, n_neurons_added[3]))
        print(' -- critic output layer V (1) --')
        

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
