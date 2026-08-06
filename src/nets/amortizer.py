import numpy as np
import torch
from einops import rearrange
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from itertools import product

from .base import NetFrame, LinearBlock
from .encoder import EncoderNet, TrialAttentionNet, TrialInvariantNet
from .invertible import InvertibleNet
from ..utils.myutils import mask_and_pad_traj_data, sort_and_pad_traj_data, get_auto_device


class AmortizerFrame(NetFrame):
    def __init__(self, config):
        super().__init__(config)

    def _set_device(self, config):
        """
        Set device for the model. 
        If device is not specified in the config, it will be automatically set.

        config (dict): Configuration dictionary for the amortizer model.
        """
        if "device" not in config:
            self.given_device = get_auto_device()
        elif config["device"] is None:
            self.given_device = get_auto_device()
        else:
            self.given_device = torch.device(config["device"])
        self.to(self.given_device)

    def infer(
        self, 
        stat_data, 
        traj_data=None, 
        n_sample=100, 
        type="mode", 
        return_samples=False,
        param_z_min=None,
        max_sample=100000
    ):
        ######################################
        """
        Infer the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        type (str, optional): Type of inference from distribution ("mode", "mean", "median"), default is "mode".
        return_samples (bool, optional): Whether to return posterior samples, default is False.
        ---
        outputs (tuple): Tuple containing inferred parameters (torch.Tensor) and posterior samples (torch.Tensor).
        """
        n_param = self.invertible_net.n_param
        if param_z_min is None:
            post_sampled, _ = self._sample(stat_data, traj_data, n_sample)
        else:
            post_sampled = list()
            num_of_sampled = 0
            for __ in range(0, max_sample, n_sample):
                _post_sampled, _ = self._sample(stat_data, traj_data, n_sample)
                for i, z in param_z_min.items():
                    _post_sampled = _post_sampled[_post_sampled[:,i] >= z]
                post_sampled.append(_post_sampled)
                num_of_sampled += len(_post_sampled)
                if num_of_sampled > n_sample: break
            if num_of_sampled == 0:
                print("No parameter set sampled ...")
                return np.array([])
            post_sampled = np.concatenate(post_sampled, axis=0)

        if type == "mode": # Maximum a Posteriori (MAP) estimation
            results = []
            # Use scipy.optimize.minimize_scalar to find the mode
            for i in range(n_param):
                kde = gaussian_kde(post_sampled[:, i])
                opt_x = minimize_scalar(
                    lambda x: -kde(x), 
                    bounds=(post_sampled[:, i].min(), post_sampled[:, i].max()), 
                    method="bounded").x
                if hasattr(opt_x, "__len__"):
                    results.append(opt_x[0])
                else:
                    results.append(opt_x)
            results = np.array(results)

        elif type == "mean":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.mean(post_sampled, axis=0)

        elif type == "median":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.median(post_sampled, axis=0)
        else:
            raise Exception(f"inappropriate type: {type}")
        
        if return_samples:
            return results, post_sampled
        else:
            return results
        



class ConditionalAmortizerFrame(AmortizerFrame):
    def infer(
        self, 
        stat_data, 
        traj_data=None, 
        n_sample=100, 
        type="mode", 
        return_samples=False,
        param_z_min=None,
        max_sample=100000
    ):
        ######################################
        """
        Infer the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        type (str, optional): Type of inference from distribution ("mode", "mean", "median"), default is "mode".
        return_samples (bool, optional): Whether to return posterior samples, default is False.
        ---
        outputs (tuple): Tuple containing inferred parameters (torch.Tensor) and posterior samples (torch.Tensor).
        """
        n_param = self.invertible_net.n_param
        if param_z_min is None:
            post_sampled, _ = self._sample(stat_data, traj_data, n_sample)
        else:
            post_sampled = list()
            num_of_sampled = 0
            for __ in range(0, max_sample, n_sample):
                _post_sampled, _ = self._sample(stat_data, traj_data, n_sample)
                for i, z in param_z_min.items():
                    _post_sampled = _post_sampled[_post_sampled[:,i] >= z]
                post_sampled.append(_post_sampled)
                num_of_sampled += len(_post_sampled)
                if num_of_sampled > n_sample: break
            if num_of_sampled == 0:
                print("No parameter set sampled ...")
                return np.array([])
            post_sampled = np.concatenate(post_sampled, axis=0)

        if type == "mode": # Maximum a Posteriori (MAP) estimation
            results = []
            # Use scipy.optimize.minimize_scalar to find the mode
            for i in range(n_param):
                kde = gaussian_kde(post_sampled[:, i])
                opt_x = minimize_scalar(lambda x: -kde(x), method="golden").x
                if hasattr(opt_x, "__len__"):
                    results.append(opt_x[0])
                else:
                    results.append(opt_x)
            results = np.array(results)

        elif type == "mean":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.mean(post_sampled, axis=0)

        elif type == "median":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.median(post_sampled, axis=0)
        else:
            raise Exception(f"inappropriate type: {type}")
        
        if return_samples:
            return results, post_sampled
        else:
            return results




class AmortizerForSummaryData(AmortizerFrame):
    def __init__(self, config=dict()):
        """
        Amortizer model for summary data (i.e., fixed-size data per inference)

        config (dict): Configuration dictionary for the amortizer model.
        """
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])
        self.invertible_net = InvertibleNet(config["invertible"])
        self._set_device(config)

    def forward(self, params, stat_data, traj_data=None):
        """
        params (ndarray): array of parameters with a shape (n_batch, n_param).
        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        if not self.encoder_net.series_data:
            params = torch.FloatTensor(params).to(self.device)
            cond = self.encoder_net(stat_data)
            out = self.invertible_net(params, cond)

        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            params = torch.FloatTensor(params).to(self.device)
            cond = self.encoder_net(stat_data, padded_traj, mask=mask)
            out = self.invertible_net(params, cond)
            
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            params = torch.FloatTensor(params)[sorted_args[-1]].to(self.device)
            cond = self.encoder_net(*sorted_args[0:-1])
            _, inv_index = sorted_args[-1].sort()
            out = self.invertible_net(params, cond)
            out = [o[inv_index] for o in out]
        return out

    def _sample(self, stat_data, traj_data=None, n_sample=100):
        """
        Sample from the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        ---
        outputs (tuple): Tuple containing posterior samples (ndarray) and log determinant of Jacobian (ndarray).
        """
        if not self.encoder_net.series_data:
            cond = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            cond = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            cond = self.encoder_net(*sorted_args[0:-1])
        args = self.invertible_net.sample(cond, n_sample)
        post_sampled, log_det_J = [arg.cpu().detach().numpy() for arg in args]
        return post_sampled, log_det_J


class AmortizerForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        """
        Amortizer model for full data from multiple i.i.d. trials (i.e., variable-size data per inference)

        config (dict): Configuration dictionary for the amortizer model.
        """
        super().__init__(config)

        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"],
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"],
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")

        self.invertible_net = InvertibleNet(config["invertible"])
        self._set_device(config)

    def forward(self, batch_param, batch_stat, batch_traj=None):
        """
        batch_param (ndarray): array of parameters with a shape (n_batch, n_param).
        batch_stat (ndarray): array of static data with a shape (n_batch, n_trial, stat_sz).
        batch_traj (list, optional): List of trajectory data, each item should have a shape (n_trial, traj_length, traj_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        full_stat_data = rearrange(batch_stat, "b e s -> (b e) s")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])
            
        else:
            full_stat_data = rearrange(batch_stat, "b e s -> (b e) s")
            full_traj_data = list()
            for data in batch_traj:
                full_traj_data += list(data)

            if self.encoder_net.traj_encoder_type == "transformer":
                padded_traj, mask = mask_and_pad_traj_data(full_traj_data)
                contexts = self.encoder_net(full_stat_data, padded_traj, mask=mask)
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])
            else:
                sorted_args = sort_and_pad_traj_data(full_stat_data, full_traj_data)
                sorted_contexts = self.encoder_net(*sorted_args[0:-1])
                _, inv_index = sorted_args[-1].sort()
                contexts = sorted_contexts[inv_index]
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])

        batch_param = torch.FloatTensor(batch_param).to(self.device)
        cond = self.trial_encoder_net(contexts)
        out = self.invertible_net(batch_param, cond)
        return out


    def _sample(self, stat_data, traj_data=None, n_sample=100):
        """
        Sample from the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, n_trial, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (n_trial, traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        ---
        outputs (tuple): Tuple containing posterior samples (ndarray) and log determinant of Jacobian (ndarray).
        """
        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            contexts = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            contexts = self.encoder_net(*sorted_args[0:-1])

        cond = self.trial_encoder_net(contexts)
        args = self.invertible_net.sample(cond, n_sample)
        post_sampled, log_det_J = [arg.cpu().detach().numpy() for arg in args]
        return post_sampled, log_det_J
    


class ConditionalAmortizerForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        super().__init__(config)

        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"]
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"]
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")

        self.invertible_net_main = InvertibleNet(config["invertible_main"])
        self.invertible_net_sub = InvertibleNet(config["invertible_sub"])
        self.dim_main_param = config["invertible_main"]["param_sz"]
        self.dim_sub_param = config["invertible_sub"]["param_sz"]

        self.post_encoder_net = TrialAttentionNet(self.dim_main_param, 
                                                  config["posterior_encoder"]["attention"]) \
            if not config["concat_feature_vector"] else None
        self.dist_sample_n = config["posterior_encoder"]["num_sample"]

        self._set_device(config)


    def forward(self, batch_param, batch_stat, batch_traj=None):
        """
        batch_param (ndarray): array of parameters with a shape (n_batch, dim_main_param + dim_sub_param * n_cond).
        batch_stat (ndarray): array of static data with a shape (n_batch, n_cond, n_trial, stat_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        full_stat_data = rearrange(batch_stat, "a b c d -> (a b c) d")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(a b c) d -> a (b c) d", a=batch_stat.shape[0], b=batch_stat.shape[1])
           
        else:
            full_stat_data = rearrange(batch_stat, "a b c d -> (a b c) d")
            traj_rearrange = rearrange(batch_traj, "a b c -> (a b c)")
            full_traj_data = list()
            for data in traj_rearrange:
                full_traj_data += list(data)
            
            if self.encoder_net.traj_encoder_type == "transformer":
                padded_traj, mask = mask_and_pad_traj_data(full_traj_data)
                contexts = self.encoder_net(full_stat_data, padded_traj, mask=mask)
                contexts = rearrange(contexts, "(a b c) d -> a (b c) d", a=batch_stat.shape[0], b=batch_stat.shape[1])
            else:
                raise NotImplementedError

        batch_main_param = torch.FloatTensor(batch_param[:,:self.dim_main_param]).to(self.device)
        batch_sub_param = torch.FloatTensor(batch_param[:,self.dim_main_param:]).to(self.device)
        batch_sub_param = rearrange(batch_sub_param, "b (c d) -> c b d", d=self.dim_sub_param)

        # Main parameter
        cond_main = self.trial_encoder_net(contexts)
        main_z, main_J = self.invertible_net_main(batch_main_param, cond_main)

        if self.post_encoder_net is not None:
            main_z_samples = torch.empty((cond_main.shape[0], self.dist_sample_n, self.dim_main_param)).to(self.device)
            for i in range(cond_main.shape[0]):
                mzs, _ = self.invertible_net_main.sample(cond_main[i:i+1], self.dist_sample_n)
                main_z_samples[i] = mzs
            encoded_main_dist = self.post_encoder_net(main_z_samples)

        # Sub parameter
        sub_z = torch.empty((batch_stat.shape[1], batch_stat.shape[0], self.dim_sub_param)).to(self.device)
        sub_J = torch.empty((batch_stat.shape[1], batch_stat.shape[0])).to(self.device)
        contexts = rearrange(contexts, "a (b c) d -> a b c d", b=batch_stat.shape[1])
        for s in range(batch_stat.shape[1]):
            cond = self.trial_encoder_net(contexts[:,s,:,:])
            if self.post_encoder_net is None:
                _sub_z, _sub_J = self.invertible_net_sub(batch_sub_param[s], torch.concat([cond_main, cond], dim=1))
            else:
                _sub_z, _sub_J = self.invertible_net_sub(batch_sub_param[s], torch.concat([encoded_main_dist, cond], dim=1))
            sub_z[s] = _sub_z
            sub_J[s] = _sub_J

        return main_z, main_J, sub_z, sub_J


    def infer(self, stat_data, traj_data=None, grouping_index=None, 
              n_sample=None, type=None, return_samples=False):
        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        else:
            raise NotImplementedError(f"{self.__name__}.infer not implemented for trajectory encoding")

        # Main parameter inference
        cond_main = self.trial_encoder_net(contexts)
        main_post_sampled, _ = self.invertible_net_main.sample(cond_main, self.dist_sample_n)
        if self.post_encoder_net is not None:
            encoded_main_dist = self.post_encoder_net(main_post_sampled)
        main_param = self._pick_param_from_posterior(main_post_sampled.cpu().detach().numpy(), 
                                                     self.dim_main_param, type)

        # Sub parameter inference
        group_index_set = torch.unique(grouping_index).cpu().detach().numpy()
        group_index_set = np.sort(group_index_set)
        sub_params = np.empty((len(group_index_set), self.dim_sub_param), dtype=np.float32)
        sub_post_sampled_list = np.empty((len(group_index_set), self.dist_sample_n, self.dim_sub_param), dtype=np.float32)
        for i, mask in enumerate(group_index_set):
            cond = self.trial_encoder_net(contexts[grouping_index == mask])
            # mp_torch = torch.from_numpy(main_param).reshape((1, -1)).to(self.device, dtype=torch.float32)
            if self.post_encoder_net is None:
                args = self.invertible_net_sub.sample(torch.concat([cond_main, cond], dim=1), self.dist_sample_n)
            else:
                args = self.invertible_net_sub.sample(torch.concat([encoded_main_dist, cond], dim=1), self.dist_sample_n)
            sub_post_sampled, _ = [arg.cpu().detach().numpy() for arg in args]
            sub_params[i] = self._pick_param_from_posterior(sub_post_sampled, self.dim_sub_param, type)
            sub_post_sampled_list[i] = sub_post_sampled
        
        if return_samples:
            return main_param, sub_params, main_post_sampled, sub_post_sampled_list
        return main_param, sub_params

    
    def _pick_param_from_posterior(self, post_sampled, n_param, type):
        if type == "mode": # Maximum a Posteriori (MAP) estimation
            results = []
            # Use scipy.optimize.minimize_scalar to find the mode
            for i in range(n_param):
                kde = gaussian_kde(post_sampled[:, i])
                opt_x = minimize_scalar(
                    lambda x: -float(kde(x)), 
                    bounds=(post_sampled[:, i].min(), post_sampled[:, i].max()), 
                    method="bounded").x
                if hasattr(opt_x, "__len__"):
                    results.append(opt_x[0])
                else:
                    results.append(opt_x)
            results = np.array(results)

        elif type == "mean":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.mean(post_sampled, axis=0)

        elif type == "median":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.median(post_sampled, axis=0)
        else:
            raise Exception(f"inappropriate type: {type}")
        return results

    
class RegressionForSummaryData(AmortizerFrame):
    def __init__(self, config=dict()):
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])
        self.linear_net = LinearBlock(**config["linear"])
        self._set_device(config)

    def forward(self, stat_data, traj_data=None):
        if not self.encoder_net.series_data:
            cond = self.encoder_net(stat_data)
            out = self.linear_net(cond)

        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            cond = self.encoder_net(stat_data, padded_traj, mask=mask)
            out = self.linear_net(cond)

        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            cond = self.encoder_net(*sorted_args[0:-1])
            _, inv_index = sorted_args[-1].sort()
            out = self.linear_net(cond)
            out = [o[inv_index] for o in out]
        return out

    def infer(self, stat_data, traj_data=None, n_sample=None, type=None, return_samples=False):
        assert not return_samples
        return self.forward(stat_data, traj_data).cpu().detach().numpy().reshape((-1,))


class RegressionForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"],
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"],
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")

        self.linear_net = LinearBlock(**config["linear"])
        self._set_device(config)

    def forward(self, batch_stat, batch_traj=None):
        full_stat_data = rearrange(batch_stat, "b e s -> (b e) s")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_stat.shape[0])
            
        else:
            full_traj_data = list()
            for data in batch_traj:
                full_traj_data += list(data)

            if self.encoder_net.traj_encoder_type == "transformer":
                padded_traj, mask = mask_and_pad_traj_data(full_traj_data)
                contexts = self.encoder_net(full_stat_data, padded_traj, mask=mask)
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_stat.shape[0])
            else:
                sorted_args = sort_and_pad_traj_data(full_stat_data, full_traj_data)
                sorted_contexts = self.encoder_net(*sorted_args[0:-1])
                _, inv_index = sorted_args[-1].sort()
                contexts = sorted_contexts[inv_index]
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_stat.shape[0])

        cond = self.trial_encoder_net(contexts)
        out = self.linear_net(cond)
        return out

    def infer(self, stat_data, traj_data=None, n_sample=None, type=None, return_samples=False):
        assert not return_samples
        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            contexts = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            contexts = self.encoder_net(*sorted_args[0:-1])
        cond = self.trial_encoder_net(contexts)
        return self.linear_net(cond).cpu().detach().numpy().reshape((-1,))
    

class ConditionalRegressionForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"],
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"],
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")
        
        self.concat_feature_vector = config.get("concat_feature_vector", False)

        self.linear_net_main = LinearBlock(**config["linear_main"])
        self.linear_net_sub = LinearBlock(**config["linear_sub"])
        self.dim_sub_param = config["linear_sub"]["out_sz"]
        self._set_device(config)


    def forward(self, batch_stat, batch_traj=None):
        full_stat_data = rearrange(batch_stat, "a b c d -> (a b c) d")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(a b c) d -> a (b c) d", a=batch_stat.shape[0], b=batch_stat.shape[1])
        elif self.encoder_net.traj_encoder_type == "transformer":
            if batch_traj is None:
                raise ValueError("Trajectory data are required by this conditional amortizer")
            full_traj_data = list(rearrange(batch_traj, "a b c -> (a b c)"))
            padded_traj, mask = mask_and_pad_traj_data(full_traj_data)
            contexts = self.encoder_net(full_stat_data, padded_traj, mask=mask)
            contexts = rearrange(
                contexts,
                "(a b c) d -> a (b c) d",
                a=batch_stat.shape[0],
                b=batch_stat.shape[1],
            )
        else:
            raise NotImplementedError(
                "Conditional regression currently supports transformer trajectories only"
            )

        cond_main = self.trial_encoder_net(contexts)
        main_param = self.linear_net_main(cond_main)

        sub_params = torch.empty((batch_stat.shape[1], batch_stat.shape[0], self.dim_sub_param)).to(self.device)
        contexts = rearrange(contexts, "a (b c) d -> a b c d", b=batch_stat.shape[1])
        for s in range(batch_stat.shape[1]):
            cond = self.trial_encoder_net(contexts[:,s,:,:])
            if self.concat_feature_vector:
                sub_params[s] = self.linear_net_sub(torch.concat([cond_main, cond], dim=1))
            else:
                sub_params[s] = self.linear_net_sub(torch.concat([main_param, cond], dim=1))
        
        sub_params = rearrange(sub_params, "s b d -> b (s d)")
        params = torch.concat([main_param, sub_params], dim=1)

        return params


    def infer(self, stat_data, traj_data=None, grouping_index=None, n_sample=None, type=None, return_samples=False):
        assert not return_samples
        assert grouping_index is not None, "You must specify how to group stat_data."

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            if traj_data is None:
                raise ValueError("Trajectory data are required by this conditional amortizer")
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            contexts = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            raise NotImplementedError(
                "Conditional regression currently supports transformer trajectories only"
            )
    
        # Main parameter inference
        cond_main = self.trial_encoder_net(contexts)
        main_param = self.linear_net_main(cond_main)

        # Sub parameter inference
        group_index_set = torch.unique(grouping_index).cpu().detach().numpy()
        group_index_set = np.sort(group_index_set)
        sub_params = torch.empty((len(group_index_set), self.dim_sub_param)).to(self.device)
        for i, mask in enumerate(group_index_set):
            cond = self.trial_encoder_net(contexts[grouping_index == mask])
            if self.concat_feature_vector:
                sub_params[i] = self.linear_net_sub(torch.concat([cond_main, cond], dim=1))
            else:
                sub_params[i] = self.linear_net_sub(torch.concat([main_param, cond], dim=1))

        return main_param.cpu().detach().numpy().reshape((-1,)), sub_params.cpu().detach().numpy()
    


class TwoLevelConditionalRegressionForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"],
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"],
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")
        
        self.concat_feature_vector = config.get("concat_feature_vector", False)

        self.linear_net_main = LinearBlock(**config["linear_main"])
        self.linear_net_sub1 = LinearBlock(**config["linear_sub1"])
        self.linear_net_sub2 = LinearBlock(**config["linear_sub2"])
        self.dim_sub1_param = config["linear_sub1"]["out_sz"]
        self.dim_sub2_param = config["linear_sub2"]["out_sz"]
        self._set_device(config)


    def forward(self, batch_stat, batch_traj=None):
        full_stat_data = rearrange(batch_stat, "mp sp1 sp2 ep dim -> (mp sp1 sp2 ep) dim")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(mp sp1 sp2 ep) dim -> mp (sp1 sp2 ep) dim", 
                                 mp=batch_stat.shape[0], 
                                 sp1=batch_stat.shape[1],
                                 sp2=batch_stat.shape[2])
            
        else:
            raise NotImplementedError(f"{self.__name__}.forward not implemented for trajectory encoding")

        # Main parameter batch
        cond_main = self.trial_encoder_net(contexts)
        main_param = self.linear_net_main(cond_main) # (mp, mp_dim)

        contexts = rearrange(contexts, "mp (sp1 sp2 ep) dim -> mp sp1 sp2 ep dim",
                             sp1=batch_stat.shape[1],
                             sp2=batch_stat.shape[2])

        # Sub1 parameter
        sub1_params = torch.empty((batch_stat.shape[1], batch_stat.shape[0], self.dim_sub1_param)).to(self.device)
        cond_sub1_list = list()
        for s1 in range(batch_stat.shape[1]):
            cond_sub1 = self.trial_encoder_net(rearrange(contexts[:,s1,:,:,:], "mp sp2 ep dim -> mp (sp2 ep) dim"))
            if self.concat_feature_vector:
                sub1_params[s1] = self.linear_net_sub1(torch.concat([cond_main, cond_sub1], dim=1))
                cond_sub1_list.append(cond_sub1)
            else:
                sub1_params[s1] = self.linear_net_sub1(torch.concat([main_param, cond_sub1], dim=1))
        

        # Sub2 parameter
        sub2_params = torch.empty((batch_stat.shape[1] * batch_stat.shape[2], batch_stat.shape[0], self.dim_sub2_param)).to(self.device)
        for s1, s2 in product(range(batch_stat.shape[1]), range(batch_stat.shape[2])):
            cond_sub2 = self.trial_encoder_net(contexts[:,s1,s2,:,:])
            if self.concat_feature_vector:
                sub2_params[s1 * batch_stat.shape[2] + s2] = \
                    self.linear_net_sub2(torch.concat([cond_main, cond_sub1_list[s1], cond_sub2], dim=1))
            else:
                sub2_params[s1 * batch_stat.shape[2] + s2] = \
                    self.linear_net_sub2(torch.concat([main_param, sub1_params[s1], cond_sub2], dim=1))
        
        
        sub1_params = rearrange(sub1_params, "sp1 mp dim -> mp (sp1 dim)")
        sub2_params = rearrange(sub2_params, "(sp1 sp2) mp dim -> mp (sp1 sp2 dim)",
                                sp1=batch_stat.shape[1],
                                sp2=batch_stat.shape[2])
        
        params = torch.concat([main_param, sub1_params, sub2_params], dim=1)
        return params


    def infer(self, stat_data, traj_data=None, cond1_index=None, cond12_index=None, **kwargs):
        assert cond1_index is not None and cond12_index is not None, "You must specify how to group stat_data."
        assert len(stat_data) == len(cond1_index) == len(cond12_index), "You must specify valid condition index information"

        cond_index_mapper = {}
        for i2, i1 in zip(cond12_index, cond1_index):
            i1, i2 = i1.item(), i2.item()
            if i2 not in cond_index_mapper.keys():
                cond_index_mapper[i2] = i1
            elif cond_index_mapper[i2] == i1:
                continue
            else:
                raise ValueError("Something wrong with cond1_index and cond12_index combination ...")

        # Per trial encoding
        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        else:
            raise NotImplementedError(f"{self.__name__}.infer not implemented for trajectory encoding")
    
        # Main parameter inference
        cond_main = self.trial_encoder_net(contexts)
        main_param = self.linear_net_main(cond_main)

        # Sub1 parameter inference
        group_index_set1 = torch.unique(cond1_index).cpu().detach().numpy()
        group_index_set1 = np.sort(group_index_set1)
        sub1_params = torch.empty((len(group_index_set1), self.dim_sub1_param)).to(self.device)
        cond_sub1_dict = dict()
        for i, mask in enumerate(group_index_set1):
            cond_sub1 = self.trial_encoder_net(contexts[cond1_index == mask])
            if self.concat_feature_vector:
                sub1_params[i] = self.linear_net_sub1(torch.concat([cond_main, cond_sub1], dim=1))
                cond_sub1_dict[mask] = cond_sub1
            else:
                sub1_params[i] = self.linear_net_sub1(torch.concat([main_param, cond_sub1], dim=1))
        
        # Sub2 parameter inference
        group_index_set2 = torch.unique(cond12_index).cpu().detach().numpy()
        group_index_set2 = np.sort(group_index_set2)
        sub2_params = torch.empty((len(group_index_set2), self.dim_sub2_param)).to(self.device)
        for i, mask in enumerate(group_index_set2):
            cond_sub2 = self.trial_encoder_net(contexts[cond12_index == mask])
            if self.concat_feature_vector:
                sub2_params[i] = self.linear_net_sub2(torch.concat([cond_main, 
                                                                   cond_sub1_dict[cond_index_mapper[mask]], 
                                                                   cond_sub2], dim=1))
            else:
                conditioned_sub1_param = sub1_params[cond_index_mapper[mask]]
                if conditioned_sub1_param.dim() == 1:
                    conditioned_sub1_param = conditioned_sub1_param.unsqueeze(0)
                sub2_params[i] = self.linear_net_sub2(torch.concat([main_param, conditioned_sub1_param, cond_sub2], dim=1))
        sub2_params = rearrange(sub2_params, "(s1 s2) dim -> s1 s2 dim", s1=len(group_index_set1))

        return (
            main_param.cpu().detach().numpy().reshape((-1,)), 
            sub1_params.cpu().detach().numpy(),
            sub2_params.cpu().detach().numpy(),
        )
