from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from .base import VariationalParameters


class GaussianVariationalParameters(VariationalParameters):
    """
    diag + low-rank Gaussian distribution.
    
    mean + N(0, exp(diag_lstd)^2) + N(0,(U.exp(rho))(U.exp(rho))^T)
    """
    
    GaussianVariationalParameter = namedtuple('GaussianVariationalParameter', \
                                          ['mean', 'diag_lstd', 'U', 'rho'])
    
    
    def __init__(self, low_rank_dimension, init_mean_routine=None, init_zero_mean=True,
                init_lstd_routine=None, init_std_level=.1, init_U_routine=None, init_U_scale=.1, eps_softabs=1e-3,
                mode='paired'):
        """
        The mean is randomized according to the specific layer randomization pattern unless init_zero_mean=True or
            init_mean_routine is provided. init_mean_routine takes precedence if not None.
        The diag_lstd is set to log(init_std_level*std(p)) where p follows the specific layer randomization pattern, unless
            init_lstd_routine is provided.
        If init_U_routine is not passed, each column of U follows the specific layer's randomization pattern, and
            rho is set to log(init_U_scale). Otherwise rho is set to 0.
            
        By default, init_num_passes is set to 1 (mean) + 1 (lstd) + low_rank_dimension.
        
        init_mean_routine: target, source, *args -> None
        init_lstd_routine: target, source, *args -> None
        init_U_routine: target, source, i, *args -> None
        init_rho_routine: target, val, i, *args -> None
        
        mode = 'single', 'paired', 'exhaustive'.
            single -> 1 sample per eps
            paired -> 2 samples per eps, symmetrized w.r.t. the mean; much better convergence properties for the mean.
            unscented -> 2*(low_rank_dimension+1) samples per eps, all possibilities of 1 to - (resp. +) all others to + (-)
            exhaustive -> 2**(low_rank_dimension+1) samples per eps, all +- combinations of eigendirections
        """
        
        super(GaussianVariationalParameters, self).__init__()       
        self.init_num_passes = 2 + low_rank_dimension        
        self.register_buffer('low_rank_dimension', Tensor([low_rank_dimension]).int())
        self.register_buffer('eps_softabs', Tensor([eps_softabs]))        
        
        if mode == 'single':
            _n_samples = 1
            _mode_int = 0
        elif mode == 'paired':
            _n_samples = 2
            _mode_int = 1
        elif mode == 'unscented':
            _n_samples = 4*low_rank_dimension
            _mode_int = 2
        elif mode == 'exhaustive':
            _n_samples = 2**(low_rank_dimension+1)
            _mode_int = 3
        else:
            raise ValueError('Mode for Gaussian Variational Inference not recognized: ' + str(mode) + \
                            '. Use single, paired [default], or exhaustive.')
        
        self.register_buffer('_num_coupled_samples', Tensor([_n_samples]).int())
        self.register_buffer('_mode', Tensor([_mode_int]).int())
        
        if mode == 'unscented':
            #_pm_lookup_u, _pm_lookup_m = self.plus_minus_tables_unscented(low_rank_dimension)
            _pm_lookup_m = torch.repeat_interleave(torch.tensor([1.,-1.]), 2*low_rank_dimension)
            _pm_lookup_u = None
        else:
            _pm_lookup_u, _pm_lookup_m = self.plus_minus_tables(_n_samples, low_rank_dimension)
        self.register_buffer('_pm_lookup_u', _pm_lookup_u)
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
            
        if init_U_routine is not None:
            self.init_U_routine = init_U_routine
            self.init_rho_routine = lambda t, v, i, *args: self._fill_parameter_data_to_lstd(t.select(-1, i), \
                                                                                             v, init_U_scale)
        else:
            self.init_U_routine = self._copy_parameter_data_to_slice
            self.init_rho_routine = lambda t, v, i, *args: self._fill_parameter_data_to_lstd(t.select(-1, i), \
                                                                                             v, init_U_scale)
            
        # caches
        self._svd_u_ia_u_cache = None
        self._logdet_sqrta_cache = None
        self._ut_isqrta_epsa_cache = None
        self._eps_u_cache = None
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
        U = Parameter(torch.stack([p]*self.low_rank_dimension.item(), dim=-1))
        rho = Parameter(torch.zeros(self.low_rank_dimension, dtype=p.dtype))
        
        return self.GaussianVariationalParameter(mean, diag_lstd, U, rho)
       
    def initialize_variational_parameter(self, vp, p, i, *args):
        if i == 0:
            self.init_mean_routine(vp.mean, p, *args)
        elif i == 1:
            self.init_lstd_routine(vp.diag_lstd, p, *args)
        else:
            i = i-2
            self.init_U_routine(vp.U, p, i, *args)
            self.init_rho_routine(vp.rho, p, i, *args)
            
        return None
    
    def sample_parameter_eps(self, vp, global_args):
        return torch.randn_like(vp.diag_lstd)
    
    def initiate_rebuild_parameters(self, global_parameters, vp_list, global_eps, eps_list):
        r"""rebuild mlog q / entropy related caches; we exploit the matrix determinant lemma and Woodbury matrix identity."""
        
        self._global_eps_cache = global_eps
        
        # compute parameter contributions
        n = len(vp_list)
        vp_contributions = [self._parameter_cache_contribution(vp_list[i], eps_list[i], self.eps_softabs) 
                            for i in torch.arange(n)]
        contrib_logdet, contrib_inv, contrib_proj, contrib_sqnorm, contrib_dim = zip(*vp_contributions)  
        
        # aggregate contributions        
        U_iA_U = torch.sum(torch.stack(contrib_inv, dim=0), dim=0)
        self._svd_u_ia_u_cache = torch.svd(U_iA_U)     
        self._logdet_sqrta_cache = torch.sum(torch.stack(contrib_logdet, dim=0))
        self._ut_isqrta_epsa_cache = torch.sum(torch.stack(contrib_proj, dim=0), dim=0)
        self._sqnorm_epsa_cache = torch.sum(torch.stack(contrib_sqnorm, dim=0))
        
        dim = torch.sum(torch.tensor(contrib_dim))
        self._dim_log_sqrt_2pi_cache = dim*(np.log(np.pi*2)/2)
        self._dim_by_2_cache = dim*0.5
        
        return            
    
    @staticmethod
    def _parameter_cache_contribution(vp, eps, small_val):
        vp_U = vp.U
        range_dm1 = list(range(vp_U.dim()-1))
        
        norm_U = torch.sqrt(torch.sum(torch.pow(vp_U, 2), dim=range_dm1) + small_val)
        V = vp_U * (torch.exp(vp.rho) / norm_U)
        
        sqrtA = torch.exp(vp.diag_lstd)
        isqrtA_V = torch.div(V, sqrtA.unsqueeze(-1))
        
        return torch.sum(vp.diag_lstd), torch.tensordot(isqrtA_V, isqrtA_V, dims=(range_dm1,range_dm1)), \
                torch.tensordot(eps, isqrtA_V, dims=eps.dim()), torch.sum(torch.pow(eps, 2)), eps.numel()
    
    def rebuild_parameter(self, vp, eps, global_args, i):  
        vp_U = vp.U
        
        norm_U = torch.sqrt(torch.sum(torch.pow(vp_U, 2), dim=list(range(vp_U.dim()-1))) + self.eps_softabs)
        V = vp_U * (torch.exp(vp.rho) / norm_U)
        
        x = global_args[i]      
        return vp.mean + torch.mul(self._pm_lookup_m[i],
                                   torch.mul(eps,torch.exp(vp.diag_lstd))*x[0] + 
                                   torch.matmul(V, x[1:].type_as(V)))
        
    @staticmethod
    def plus_minus_tables(n, d):
        """
        Look-up table to know if we have to switch the sign of eps along one of the low-rank dimensions
        """        
        if n==1:
            return torch.ones(1, d, dtype=torch.int32), torch.ones(1, dtype=torch.int32)        
        
        # we want rows 0,1 to be opposite of each others
        pm_lookup = np.unpackbits(np.arange(n,dtype=np.uint8).reshape(-1,1),axis=1,count=d+1,bitorder='little').astype(np.int)
        pm_lookup = 1 - 2*pm_lookup                                    
        return torch.from_numpy(pm_lookup[:,1:]), torch.from_numpy(pm_lookup[:,0])
    
    @staticmethod
    def plus_minus_tables_unscented(d):
        # obsolete
        pm_lookup = torch.ones(d+1,d+1, dtype=torch.int32)
        pm_lookup[0,1:] = -1
        pm_lookup.reshape(-1)[d+2::d+2] = -1
        pm_lookup = np.c_[pm_lookup,pm_lookup]
        pm_lookup[:,d+1] = -1
        pm_lookup = pm_lookup.reshape(-1,d+1)
        return torch.from_numpy(pm_lookup[:,1:]), torch.from_numpy(pm_lookup[:,0])
    
    def sample_globals(self, global_parameters):
        """
        Each row gives the linear combination of [diag(lstd)*eps, U_1, U_2, ...] for one of the (correlated) samples
        """
        if self._mode == 2: # unscented
            # sample a rotation matrix
            M = torch.randn(self.low_rank_dimension, self.low_rank_dimension, device=self.low_rank_dimension.device)
            R = torch.svd(M, some=False).U
            
            # sample the norm with a naive chi2
            z = torch.sqrt(
                torch.sum(
                    torch.pow(
                        torch.randn(self.low_rank_dimension, device=self.low_rank_dimension.device), 2)))
            
            # linear combinations are pairwise orthogonal and each one results in the correct covariance (in expectation)
            eps_U = R*z
            
            # taking +- diag(lstd)*eps_A
            g_eps_A = torch.repeat_interleave(torch.tensor([1.,-1.], device=self.low_rank_dimension.device), 
                                              repeats=self.low_rank_dimension.item())
            eps = torch.cat([g_eps_A.unsqueeze(-1), eps_U.repeat(2,1)], dim=-1)
            
            # +- combinations around the mean
            return eps.repeat(2,1)
        else:
            eps_U = torch.randn(self.low_rank_dimension, device=self.low_rank_dimension.device)
            x_U = torch.mul(self._pm_lookup_u, eps_U) 
            return torch.cat([torch.ones(self._num_coupled_samples, 1), x_U], dim=1)
    
    def mlog_q(self, global_parameters, vp_list, p_list, i):
        # we shortcut computations directly in terms of eps, using the Woodbury matrix identity.
        # allows to reuse much of the computations for all samples without having to cache much either.
        
        # the +- overall sign from _pm_lookup_m does not matter for energy computations
        global_eps_sample = self._global_eps_cache[i] # [0] applied to the diag part, [1:] to principal subspace
        eps_A = global_eps_sample[0] # it's just +-1 at the time, just leaving it here and below in case.       
        eps_U = global_eps_sample[1:]
        
        x_A = self._ut_isqrta_epsa_cache * eps_A
        
        svd = self._svd_u_ia_u_cache
        proj_eps_U = torch.mv(svd.U.t(), eps_U.type_as(svd.U))
        proj_x_A = torch.mv(svd.U.t(), x_A)
        S_plus_1 = 1 + svd.S
        
        E_uu = torch.sum(torch.pow(proj_eps_U,2) * svd.S / S_plus_1)
        E_aa = self._sqnorm_epsa_cache * torch.pow(eps_A,2) - torch.sum(torch.pow(proj_x_A, 2) / S_plus_1)
        E_ua = torch.sum(proj_x_A * proj_eps_U / S_plus_1)
        
        E = (E_uu + E_aa)*.5 + E_ua  
        return E + self.normalisation()
    
    def entropy_q(self, global_parameters, vp_list):
        return self.normalisation() + self._dim_by_2_cache
    
    def normalisation(self):
        return self._logdet_sqrta_cache + torch.sum(torch.log1p(self._svd_u_ia_u_cache.S))*.5 + self._dim_log_sqrt_2pi_cache
    
    def variational_parameter_names(self):
        return self.GaussianVariationalParameter._fields
    
    def variational_parameter_type(self):
        return self.GaussianVariationalParameter
    
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