from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from .base import VariationalParameters


class DiagonalGaussianVariationalParameters(VariationalParameters):
    """
    diag Gaussian distribution.
    
    mean + N(0, exp(diag_lstd)^2)
    """
    
    DiagonalGaussianVariationalParameter = namedtuple('DiagonalGaussianVariationalParameter', \
                                                      ['mean', 'diag_lstd'])
    
    
    def __init__(self, init_mean_routine=None, init_zero_mean=True,
                init_lstd_routine=None, init_std_level=.1,
                mode='paired'):
        """
        The mean is randomized according to the specific layer randomization pattern unless init_zero_mean=True or
            init_mean_routine is provided. init_mean_routine takes precedence if not None.
        The diag_lstd is set to log(init_std_level*std(p)) where p follows the specific layer randomization pattern, unless
            init_lstd_routine is provided.
            
        By default, init_num_passes is set to 1 (mean) + 1 (lstd).
        
        init_mean_routine: target, source, *args -> None
        init_lstd_routine: target, source, *args -> None
        
        mode = 'single', 'paired'.
            single -> 1 sample per eps
            paired -> 2 samples per eps, symmetrized w.r.t. the mean; much better convergence properties for the mean.
        """
        
        super(DiagonalGaussianVariationalParameters, self).__init__()       
        self.init_num_passes = 2       
        
        if mode == 'single':
            _n_samples = 1
            _mode_int = 0
        elif mode == 'paired':
            _n_samples = 2
            _mode_int = 1
        else:
            raise ValueError('Mode for Diagonal Gaussian Variational Inference not recognized: ' + str(mode) + \
                            '. Use single or paired [default].')
        
        self.register_buffer('_num_coupled_samples', Tensor([_n_samples]).int())
        self.register_buffer('_mode', Tensor([_mode_int]).int())
        
        _pm_lookup_m = torch.tensor([1, -1], dtype=torch.int32)
        self.register_buffer('_pm_lookup_m', _pm_lookup_m)
        
        if init_mean_routine is not None:
            self.init_mean_routine = init_mean_routine
        elif init_zero_mean:
            self.init_mean_routine = self._zero_parameter_data
        else:
            self.init_mean_routine = self._copy_parameter_data
            
        if init_lstd_routine is not None:
            self.init_lstd_routine = init_lstd_routine
        else:
            self.init_lstd_routine = lambda t, s, *args: self._fill_parameter_data_to_lstd(t, s, init_std_level)
            
        # caches
        self._logdet_sqrta_cache = None
        self._sqnorm_epsa_cache = None
        self._dim_log_sqrt_2pi_cache = None
        self._dim_by_2_cache = None
    
    def num_coupled_samples(self):
        """
        For some distributions, inference works much better when using a tuple of joint samples or more.
        In that case, the Variationalize module needs to know in advance that it needs to keep track of more than one model.
        """
        return self._num_coupled_samples.item()
    
    def _to_variational_parameter(self, p):
        mean = Parameter(torch.empty_like(p))
        diag_lstd = Parameter(torch.empty_like(p))
        
        return self.DiagonalGaussianVariationalParameter(mean, diag_lstd)
       
    def initialize_variational_parameter(self, vp, p, i, *args):
        if i == 0:
            self.init_mean_routine(vp.mean, p, *args)
        elif i == 1:
            self.init_lstd_routine(vp.diag_lstd, p, *args)
        else:
            raise ValueError("Expected i=0 or 1, received {}.".format(i))
            
        return None
    
    def sample_parameter_eps(self, vp, global_args):
        return torch.randn_like(vp.diag_lstd)
    
    def initiate_rebuild_parameters(self, global_parameters, vp_list, global_eps, eps_list):
        r"""rebuild mlog q / entropy related caches; we exploit the matrix determinant lemma and Woodbury matrix identity."""
        
        # compute parameter contributions
        n = len(vp_list)
        vp_contributions = [self._parameter_cache_contribution(vp_list[i], eps_list[i]) 
                            for i in torch.arange(n)]
        contrib_logdet, contrib_sqnorm, contrib_dim = zip(*vp_contributions)  
        
        # aggregate contributions        
        self._logdet_sqrta_cache = torch.sum(torch.stack(contrib_logdet, dim=0))
        self._sqnorm_epsa_cache = torch.sum(torch.stack(contrib_sqnorm, dim=0))
        
        dim = torch.sum(torch.tensor(contrib_dim))
        self._dim_log_sqrt_2pi_cache = dim*(np.log(np.pi*2)/2)
        self._dim_by_2_cache = dim*0.5
        
        return            
    
    @staticmethod
    def _parameter_cache_contribution(vp, eps):
        return torch.sum(vp.diag_lstd), torch.sum(torch.pow(eps, 2)), eps.numel()
    
    def rebuild_parameter(self, vp, eps, global_args, i):        
        return vp.mean + torch.mul(self._pm_lookup_m[i],
                                   torch.mul(eps,torch.exp(vp.diag_lstd)))
    
    def sample_globals(self, global_parameters):
        return None
    
    def mlog_q(self, global_parameters, vp_list, p_list, i):
        # we shortcut computations directly in terms of eps.
        # allows to reuse much of the computations for all samples without having to cache much either.
        
        E = self._sqnorm_epsa_cache*.5 
        return E + self.normalisation() 
    
    def entropy_q(self, global_parameters, vp_list):
        return self.normalisation() + self._dim_by_2_cache
    
    def normalisation(self):
        return self._logdet_sqrta_cache + self._dim_log_sqrt_2pi_cache
    
    def variational_parameter_names(self):
        return self.DiagonalGaussianVariationalParameter._fields
    
    def variational_parameter_type(self):
        return self.DiagonalGaussianVariationalParameter
    
    def num_passes_initialize(self):       
        return self.init_num_passes    
            
    @staticmethod
    def _fill_parameter_data_to_lstd(target, source, *args):
        """
        Utility function for the initialization of variational parameters.
        """
            
        with torch.no_grad():
            val = torch.sqrt(torch.mean(torch.pow(source,2)))
            if val==0.:
                val.data.fill_(1e-3)
            if args:
                val = val*args[0]
            val.log_() 
            target.data.fill_(val) 