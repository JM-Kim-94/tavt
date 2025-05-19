"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm

from torch.nn.utils import spectral_norm

def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)



class MyFlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """
    def forward(self, *inputs, **kwargs):  # o, a, c = ([128,20]), ([128,6]), ([128,5])
        flat_inputs = torch.cat(inputs, dim=1)  # ([128,31])
        emb = super().forward(flat_inputs, **kwargs)  # 임베딩 = ([128,20])
        q_value = emb.sum(dim=1).unsqueeze(1)  # Q 밸류 = ([128, 1])
        return emb, q_value





class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)






class TransitionDecoder4PEARL(nn.Module):
    def __init__(self, task_latent_dim, obs_dim, state_dim, action_dim, reward_dim):
        super().__init__()
        self.task_latent_dim = task_latent_dim
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.LogVar_Min = -20
        self.LogVar_Max = 2

        """ #################### 디코더 #################### """
        """ state embed (phi) """
        self.state_emb_fc = nn.Sequential(nn.Linear(obs_dim, 200), nn.ReLU(),
                                          nn.Linear(200, 200), nn.ReLU(),
                                          # nn.Linear(200, 200), nn.ReLU(),
                                          nn.Linear(200, state_dim))

        self.base_fc = nn.Sequential(nn.Linear(task_latent_dim + state_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU())

        """ obs, action 배치 디코더 """
        self.obs_action_dec = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                            nn.Linear(256, 256), nn.ReLU(),
                                            nn.Linear(256, obs_dim + 2 * action_dim))

        """ reward, next_obs 디코더 """
        self.reward_next_obs_term_dec = nn.Sequential(nn.Linear(256 + action_dim, 256), nn.ReLU(),
                                                      nn.Linear(256, 256), nn.ReLU(),
                                                      nn.Linear(256, 2 * obs_dim + 2))
    def state_embed(self, obs):
        state = self.state_emb_fc(obs)
        return state

    def dec1(self, c, s):
        base_input = torch.cat([c, s], dim=-1)  # ([16, 128, 20])
        h1 = self.base_fc(base_input)  # ([16, 128, 256])
        obs_action_dec_out = self.obs_action_dec(h1)  # ([16, 128, 32])

        """ obs pred """  # --> probabilistic --> deterministic
        obs_pred = obs_action_dec_out[:, :, : self.obs_dim]  # ([16, 128, 20])

        """ action pred """  # --> probabilistic
        action_pred_mean = obs_action_dec_out[:, :, self.obs_dim: self.obs_dim + self.action_dim]
        action_pred_logvar = obs_action_dec_out[:, :, self.obs_dim + self.action_dim:]

        action_pred_logvar = torch.clamp(action_pred_logvar, self.LogVar_Min, self.LogVar_Max)
        action_pred_std = torch.exp(action_pred_logvar)
        eps = torch.randn_like(action_pred_std)
        action_pred_sample = eps.mul(action_pred_std).add_(action_pred_mean)

        return obs_pred, action_pred_sample


    def dec2(self, c, s, a):
        base_input = torch.cat([c, s], dim=-1)
        h1 = self.base_fc(base_input)
        h2 = torch.cat([h1, a], dim=-1)

        reward_next_obs_dec_out = self.reward_next_obs_term_dec(h2)

        """ reward 프리딕션 """
        reward_pred = reward_next_obs_dec_out[:, :, 0].unsqueeze(-1)  # [16, 128, 1]

        """ next_obs 프리딕션 """
        next_obs_pred_mean = reward_next_obs_dec_out[:, :, 1: self.obs_dim + 1]
        next_obs_pred_logvar = reward_next_obs_dec_out[:, :, self.obs_dim + 1: -1]

        next_obs_pred_logvar = torch.clamp(next_obs_pred_logvar, self.LogVar_Min, self.LogVar_Max)
        next_obs_pred_std = torch.exp(next_obs_pred_logvar)
        eps = torch.randn_like(next_obs_pred_std)
        next_obs_pred = eps.mul(next_obs_pred_std).add_(next_obs_pred_mean)

        """ terminal 프리딕션 """
        term_pred = reward_next_obs_dec_out[:, :, -1].unsqueeze(-1)
        term_pred = torch.sigmoid(term_pred)

        return reward_pred, next_obs_pred, next_obs_pred_mean, next_obs_pred_std, term_pred


    def forward(self, c, s, a):
        pass



class WGanCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim, latent_dim):
        super().__init__()
        self.input_dim = 2 * obs_dim + action_dim + reward_dim + latent_dim

        self.fc = nn.Sequential(nn.Linear(self.input_dim, 200), nn.ReLU(),
                                nn.Linear(200, 200), nn.ReLU(),
                                nn.Linear(200, 200), nn.ReLU(),
                                nn.Linear(200, 1))

    def forward(self, samples):
        # inputs = samples.view(-1, self.input_dim)
        critic_score = self.fc(samples)
        return critic_score





class Decoder_old(nn.Module):  # 11.1 이전까지 사용
    def __init__(self, latent_dim, obs_dim, action_dim, reward_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.logvar_min = -20
        self.logvar_max = 2

        self.dec1_fc = nn.Sequential(nn.Linear(obs_dim + latent_dim, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 2 * action_dim))

        self.decoder_fc = nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 2 * obs_dim + 1))

    def dec1(self, o, c):
        sample = torch.cat([o, c], dim=-1)
        pred = self.dec1_fc(sample)

        a_pred_mean   = pred[:, : self.action_dim]  # ([128, 20])
        a_pred_logvar = pred[:, self.action_dim :]  # ([128, 20])

        a_pred_logvar = torch.clamp(a_pred_logvar, self.logvar_min, self.logvar_max)
        a_pred_std    = torch.exp(a_pred_logvar * 0.5)  # logval = log(std^2)  --> std = exp(0.5*logvar)

        a_eps = torch.randn_like(a_pred_std)
        a_pred_sample = a_eps.mul(a_pred_std).add_(a_pred_mean)

        return a_pred_sample

    def forward(self, o, a, c):
        sample = torch.cat([o, a, c], dim=-1)
        pred = self.decoder_fc(sample)

        r_pred             = pred[:, 0].unsqueeze(-1)  # ([128, 1])
        next_o_mean_pred   = pred[:, self.reward_dim : self.reward_dim + self.obs_dim]  # ([128, 20])
        next_o_logvar_pred = pred[:, self.reward_dim + self.obs_dim : self.reward_dim + self.obs_dim + self.obs_dim]  # ([128, 20])

        next_o_logvar_pred = torch.clamp(next_o_logvar_pred, self.logvar_min, self.logvar_max)
        next_o_std_pred    = torch.exp(next_o_logvar_pred * 0.5)  # logval = log(std^2)  --> std = exp(0.5*logvar)

        next_o_eps = torch.randn_like(next_o_std_pred)
        next_o_sample_pred = next_o_eps.mul(next_o_std_pred).add_(next_o_mean_pred)

        return r_pred, next_o_sample_pred, next_o_mean_pred, next_o_std_pred




# k_decoder = Decoder(10, 29, 9, 1, False, False)
# c_decoder = Decoder(10, 29, 9, 1, False, True)
# c_decoder = Decoder(
#         latent_dim=latent_dim,
#         obs_dim=obs_dim,
#         action_dim=action_dim,
#         reward_dim=1,
#         use_next_state=use_decoder_next_state,
#         use_k=False,
#         num_tasks=num_train,
#         use_state_noise=use_state_noise
#     )


class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, action_dim, reward_dim, use_next_state, use_k, num_tasks, use_state_noise):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        # self.logvar_min = -20
        # self.logvar_max = 2

        self.use_next_state = use_next_state
        self.use_k = use_k
        self.use_state_noise = use_state_noise

        if self.use_k:
            self.embed = nn.Sequential(nn.Linear(num_tasks, latent_dim), nn.ReLU())
            self.decoder_fc_base = nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.ReLU())            
        else:
            self.decoder_fc_base = nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, 256), nn.ReLU(),
                                                nn.Linear(256, 256), nn.ReLU(),
                                                nn.Linear(256, 256), nn.ReLU())
            

        if self.use_next_state:
            if self.use_state_noise:
                self.decoder_fc_out = nn.Linear(256, 2 * obs_dim + 1)
            else:
                self.decoder_fc_out = nn.Linear(256, obs_dim + 1)
        else:
            self.decoder_fc_out = nn.Linear(256, 1) 

    def forward(self, o, a, c):
        if self.use_k:
            c = self.embed(c)
        sample = torch.cat([o, a, c], dim=-1)        
        pred = self.decoder_fc_base(sample)
        pred = self.decoder_fc_out(pred)

        if self.use_next_state:
            if self.use_state_noise:
                r_pred             = pred[:, 0].unsqueeze(-1)  # ([128, 1])
                next_o_mean_pred   = pred[:, self.reward_dim : self.reward_dim + self.obs_dim]  # ([128, 20])
                next_o_logvar_pred = pred[:, self.reward_dim + self.obs_dim : self.reward_dim + self.obs_dim + self.obs_dim]  # ([128, 20])

                # next_o_logvar_pred = torch.clamp(next_o_logvar_pred, self.logvar_min, self.logvar_max)
                next_o_std_pred    = torch.exp(next_o_logvar_pred * 0.5)  # logvar = log(std^2)  --> std = exp(0.5*logvar)

                next_o_eps = torch.randn_like(next_o_std_pred)
                # next_o_sample_pred = next_o_eps.mul(next_o_std_pred).add_(next_o_mean_pred)
                next_o_sample_pred = next_o_mean_pred + next_o_eps * next_o_std_pred

                return r_pred, next_o_sample_pred, next_o_mean_pred, next_o_std_pred  #

            else:
                r_pred             = pred[:, 0].unsqueeze(-1)  # ([128, 1])
                next_o_mean_pred   = pred[:, self.reward_dim : self.reward_dim + self.obs_dim]  # ([128, 20])
                next_o_std_pred    = torch.zeros_like(next_o_mean_pred)

                return r_pred, next_o_mean_pred, next_o_mean_pred, next_o_std_pred
        
        else:
            return pred, None, None, None


class AlphaNet(nn.Module):
    def __init__(self, latent_dim):
        """
        input : task latent variable 'c',
        output : log_alpha,
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, c):
        h = F.relu(self.fc1(c))
        h = F.relu(self.fc2(h))
        log_alpha = self.fc3(h)
        return log_alpha


class ReverseDynamics(nn.Module):
    def __init__(self, latent_dim, obs_dim, action_dim, reward_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.logvar_min = -20
        self.logvar_max = 2

        """backward"""
        # input = s', r, z
        # self.decoder_fc_base = nn.Sequential(nn.Linear(obs_dim + reward_dim + latent_dim, 256), nn.ReLU(),
        #                                      nn.Linear(256, 256), nn.ReLU(),
        #                                      nn.Linear(256, 256), nn.ReLU())
        # # output = s, a
        # self.decoder_fc_out = nn.Linear(256, obs_dim + action_dim)

        """forward"""
        # input = s, a, z
        self.decoder_fc_base = nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, 256), nn.ReLU(),
                                             nn.Linear(256, 256), nn.ReLU(),
                                             nn.Linear(256, 256), nn.ReLU())
        # output = r, s'
        self.decoder_fc_out = nn.Linear(256, 1 + obs_dim)

    """backward"""
    # def forward(self, next_obs, r, z):
    #     sample = torch.cat([next_obs, r, z], dim=-1)
    #     pred = self.decoder_fc_base(sample)
    #     pred = self.decoder_fc_out(pred)
    #
    #     obs_pred = pred[:, : self.obs_dim]  # ([128, 20])
    #     a_pred = pred[:, self.obs_dim: self.obs_dim + self.action_dim]  # ([128, 6])
    #
    #     return obs_pred, a_pred

    """forward"""
    def forward(self, obs, a, z):
        sample = torch.cat([obs, a, z], dim=-1)
        pred = self.decoder_fc_base(sample)
        pred = self.decoder_fc_out(pred)

        r_pred = pred[:, 0].unsqueeze(-1)  # ([128, 1])
        next_obs_pred = pred[:, self.reward_dim : self.reward_dim + self.obs_dim]  # ([128, 20])

        return r_pred, next_obs_pred




class PsiAuxVaeDec(nn.Module):
    def __init__(self, latent_dim, obs_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.logvar_min = -20
        self.logvar_max = 2

        # input = s, a, z
        self.decoder_fc_base = nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, 256), nn.ReLU(),
                                             nn.Linear(256, 256), nn.ReLU(),
                                             nn.Linear(256, 256), nn.ReLU())
        self.decoder_fc_out = nn.Linear(256, obs_dim + action_dim)

    def forward(self, obs, a, z):
        sample = torch.cat([obs, a, z], dim=-1)
        pred = self.decoder_fc_base(sample)
        pred = self.decoder_fc_out(pred)

        obs_recon = pred[:, : self.obs_dim]  # ([128, 1])
        action_recon = pred[:, self.obs_dim: self.obs_dim + self.action_dim]  # ([128, 6])

        return obs_recon, action_recon







class Discriminator(nn.Module):  # standard GAN
    def __init__(self, obs_dim, action_dim, reward_dim):
        super().__init__()
        self.input_dim = 2 * obs_dim + action_dim + reward_dim

        self.fc = nn.Sequential(nn.Linear(self.input_dim, 200), nn.ReLU(),
                                nn.Linear(200, 200), nn.ReLU(),
                                # nn.Linear(200, 200), nn.ReLU(),
                                nn.Linear(200, 1), nn.Sigmoid())

    def forward(self, samples):
        # inputs = samples.view(-1, self.input_dim)
        logit = self.fc(samples)
        return logit




class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.enc_fc = nn.Sequential(nn.Linear(input_dim, 200), nn.ReLU(),
                                    nn.Linear(200, 200), nn.ReLU(),
                                    nn.Linear(200, latent_dim))

        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, 200), nn.ReLU(),
                                    nn.Linear(200, 200), nn.ReLU(),
                                    nn.Linear(200, output_dim))

    def forward(self, task_c):
        z = self.enc_fc(task_c)
        task_c_recon = self.dec_fc(z)

        return z, task_c_recon



class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, 100)
        self.enc_fc2 = nn.Linear(100, 100)
        self.enc_fc_mu = nn.Linear(100, latent_dim)
        self.enc_fc_logvar = nn.Linear(100, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 100)
        self.dec_fc2 = nn.Linear(100, 100)
        self.dec_fc_out = nn.Linear(100, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.enc_fc1(x))
        h = self.relu(self.enc_fc2(h))
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        x = self.dec_fc_out(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



# Spectral Normalization

# https://gh.mlsub.net/kwotsin/mimicry/blob/master/torch_mimicry/modules/spectral_norm.py
class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """
    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer('sn_u', torch.randn(1, n_dim))
        self.register_buffer('sn_sigma', torch.ones(1))

    @property
    def u(self):
        return getattr(self, 'sn_u')

    @property
    def sigma(self):
        return getattr(self, 'sn_sigma')

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(W=W,
                                            u=self.u,
                                            num_iters=self.num_iters,
                                            eps=self.eps)

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma

class ManualSNLinear(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_features,
                              num_iters=kwargs.get('num_iters', 1))
    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


def SNLinear(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if default:
        return torch.nn.utils.spectral_norm(nn.Linear(*args, **kwargs))  # 원래 이걸로 돼있었음
        # return torch.nn.utils.parametrizations.spectral_norm(nn.Linear(*args, **kwargs))
    else:
        # https://gh.mlsub.net/kwotsin/mimicry/blob/master/torch_mimicry/modules/spectral_norm.py 에서 구현한 모듈
        return spectral_norm.ManualSNLinear(*args, **kwargs)


class SNWGanCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim, latent_dim):
        super().__init__()
        self.input_dim = 2 * obs_dim + action_dim + reward_dim + latent_dim

        self.fc = nn.Sequential(SNLinear(self.input_dim, 200), nn.ReLU(),
                                SNLinear(200, 200), nn.ReLU(),
                                SNLinear(200, 200), nn.ReLU(),
                                SNLinear(200, 1))

    def forward(self, samples):
        # inputs = samples.view(-1, self.input_dim)
        critic_score = self.fc(samples)
        return critic_score








