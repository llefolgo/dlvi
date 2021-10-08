from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
from torch.nn.parameter import Parameter
from torch import Tensor

from .base import VariationalParameters


class GMMVariationalParameters(VariationalParameters):
    """
    diag + low-rank Gaussian distribution per component of the mixture.
    
    mean + N(0, exp(diag_lstd)^2) + N(0,(U.exp(rho))(U.exp(rho))^T)
    
    The (global) mixture proportions pi use a softmax parametrization pi_k = exp(alpha_k)/sum exp(alpha_k')
    """
    
    GMMVariationalParameter = namedtuple('GMMVariationalParameter', \
                                          ['mean', 'diag_lstd', 'U', 'rho'])
    GMMConcentration = namedtuple('GMMConcentration', ['alpha'])
    
    
    def __init__(self, low_rank_dimension, num_components, 
                 init_mean_routine=None, init_zero_mean=True, init_lstd_routine=None, 
                 init_std_level=.1, init_U_routine=None, init_U_scale=.1, eps_softabs=1e-3,
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
            
        In the sampling scheme, for now I'm opting to take the same eps for all components, so that we don't get a much more
        unlucky/unlikely draw for one than for the other. Might be preferable given that we also optimize the mixture
        proportions? 
        The natural variant is commented.
        """
        
        super(GMMVariationalParameters, self).__init__()       
        self.init_num_passes = (2 + low_rank_dimension) * num_components # mean,lstd,U * num_components       
        self.register_buffer('low_rank_dimension', Tensor([low_rank_dimension]).int())
        self.register_buffer('num_components', Tensor([num_components]).int())
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
            raise ValueError('Mode for GMM Variational Inference not recognized: ' + str(mode) + \
                            '. Use single, paired [default], or exhaustive.')
        
        self.register_buffer('_num_samples_per_component', Tensor([_n_samples]).int())
        self.register_buffer('_mode', Tensor([_mode_int]).int())
        
        if mode == 'unscented':
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
            self.init_rho_routine = lambda t, v, i, *args: self._copy_parameter_data_to_slice(t, Tensor([0.]), i)
        else:
            self.init_U_routine = self._copy_parameter_data_to_slice
            self.init_rho_routine = lambda t, v, i, *args: self._copy_parameter_data_to_slice(t, \
                                                                      torch.log(Tensor([init_U_scale])), i)
            
        # caches
        self.sqrt_schur_complement_cache = None       
        self._logdet_cache = None
        self._dim_log_sqrt_2pi_cache = None
        self._isqrtA_cache = None
        self._isqrtA_V_cache = None
    
    def num_coupled_samples(self):
        """
        For some distributions, inference works much better when using a tuple of joint samples or more.
        In that case, the Variationalize module needs to know in advance that it needs to keep track of more than one model.
        """
        return (self._num_samples_per_component*self.num_components).item()
    
    def coupled_sample_weights(self, global_parameters, global_eps, device=None):
        # has to be consistent with which model implements which component / rank
        # (sample 1 for comp 1), ..., (sample 1 for comp K), (sample 2 for comp 1), etc. 
        pi = softmax(global_parameters.alpha, dim=0, _stacklevel=5)
        return pi.repeat(self._num_samples_per_component.item()) / self._num_samples_per_component
    
    def global_parameters(self):
        """
        alpha
        """
        alpha = Parameter(torch.full((self.num_components.item(),), 
                                     -torch.log(self.num_components.float()).item())
                         )
        return self.GMMConcentration(alpha) 
    
    def _to_variational_parameter(self, p):
        # prepend components index [, append principal subspace dir index]
        tmp = torch.stack([p]*self.num_components.item(), dim=0)
        
        mean = Parameter(tmp.clone())
        diag_lstd = Parameter(tmp.clone())
        U = Parameter(torch.stack([tmp]*self.low_rank_dimension.item(), dim=-1))
        rho = Parameter(torch.zeros(self.num_components, self.low_rank_dimension, dtype=p.dtype))
        
        return self.GMMVariationalParameter(mean, diag_lstd, U, rho)
       
    def initialize_variational_parameter(self, vp, p, i, *args):
        # i ranges from 0 to (2+lr)*num_comp excluded
        j, k = divmod(i, self.num_components.item())
        
        if j == 0:
            self.init_mean_routine(vp.mean[k], p, *args)
        elif j == 1:
            self.init_lstd_routine(vp.diag_lstd[k], p, *args)
        else:
            j = j-2
            self.init_U_routine(vp.U[k], p, j, *args)
            self.init_rho_routine(vp.rho[k], None, j, *args)
            
        return None
    
    def sample_parameter_eps(self, vp, global_args):
        return torch.randn_like(vp.diag_lstd[0])
        
    def initiate_rebuild_parameters(self, global_parameters, vp_list, global_eps, eps_list):
        r"""rebuild mlog q / entropy related caches; we exploit the matrix determinant lemma and Woodbury matrix identity."""
        
        # compute parameter contributions
        n = len(vp_list)
        vp_contributions = [self._parameter_cache_contribution(vp_list[i], eps_list[i], self.eps_softabs) 
                            for i in torch.arange(n)]
        contrib_logdet, contrib_inv, contrib_isqrt_diag, contrib_lr_proj, contrib_dim = zip(*vp_contributions)  
        
        # aggregate contributions        
        U_iA_U = torch.sum(torch.stack(contrib_inv, dim=0), dim=0)
        svd = torch.svd(U_iA_U) # batched mode
        
        self.sqrt_schur_complement_cache = svd.U.transpose(-2, -1) / torch.sqrt(1 + svd.S).unsqueeze(-1) 
            # batching prevents simpler expressions like (svd.U / torch.sqrt(1 + svd.S)).transpose(-2,-1)
        self._logdet_cache = torch.sum(torch.stack(contrib_logdet, dim=0), dim=0) + torch.sum(torch.log1p(svd.S), dim=-1)*.5
        self._dim_log_sqrt_2pi_cache = torch.sum(torch.tensor(contrib_dim))*(np.log(np.pi*2)/2)
        
        # store the rest for lack of a better option
        self._isqrtA_cache = contrib_isqrt_diag
        self._isqrtA_V_cache = contrib_lr_proj
        
        return            
    
    @staticmethod
    def _parameter_cache_contribution(vp, eps, small_val):
        vp_U = vp.U
        range_dm1 = list(range(1, vp_U.dim()-1)) # dimension 0 is the GMM component dim, last dimension is the principal mode's
        
        norm_U = torch.sqrt(torch.sum(torch.pow(vp_U, 2), dim=range_dm1, keepdim=True) + small_val)
        V = vp_U * (torch.exp(vp.rho).view(norm_U.size()) / norm_U)
        
        isqrtA = torch.exp(-vp.diag_lstd)
        isqrtA_V = torch.mul(V, isqrtA.unsqueeze(-1))
        
        # einsum doesn't allow for ellipsis reduction, and torch.einsum doesn't provide numpy's explicit mode
        # tensordot doesn't have a batched mode.
        # Here is perhaps the most beautiful piece of code of this file ;|
        ellipsis = 'abcdefghlmnopqrstuvwxyz'[0:(isqrtA_V.dim()-2)]
        formula = 'k' + ellipsis + 'i,' + 'k' + ellipsis + 'j->kij'
        
        return torch.sum(vp.diag_lstd, dim=range_dm1), torch.einsum(formula, isqrtA_V, isqrtA_V), \
                isqrtA, isqrtA_V, eps.numel()
    
    def rebuild_parameter(self, vp, eps, global_args, i):
        # i ranges from 0 to num_models = num_samples_per_comp*num_comp excluded
        j, k = divmod(i, self.num_components.item())
        
        vp_U = vp.U[k]        
        norm_U = torch.sqrt(torch.sum(torch.pow(vp_U, 2), dim=list(range(vp_U.dim()-1))) + self.eps_softabs)
        V = vp_U * (torch.exp(vp.rho[k]) / norm_U)
        
        x = global_args[j]      
        return vp.mean[k] + torch.mul(self._pm_lookup_m[j],
                                   torch.mul(eps,torch.exp(vp.diag_lstd[k]))*x[0] + 
                                   torch.matmul(V, x[1:]))
    
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
    
    def sample_globals(self, global_parameters):
        """
        Each row gives the linear combination of [diag(lstd), U_1, U_2, ...] for one of the (correlated) samples
        Using the same eps across all GMM components for now
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
            return torch.cat([torch.ones(self._num_samples_per_component, 1), x_U], dim=1)
        
    def mlog_q(self, global_parameters, vp_list, p_list, i):
        # The computations use the Woodbury matrix identity.
        # We reuse some of the computations for all samples.
        # The implementation is different than from a single component (Gaussian) model as the shortcut
        # directly in terms of eps is not available. 
        
        # The code is tougher to follow because we have the first dimension as the GMM dimension 
        # that we can't just collapse until the end
        tuple_list = zip(self._isqrtA_cache, p_list, vp_list)
        isqrtA_p_list = [torch.mul(tple[1].unsqueeze(0) - tple[2].mean, tple[0]) for tple in tuple_list]
        
        K = self.num_components.item()
        E_diag_list = [torch.einsum('ke,ke->k', v.view(K,-1), v.view(K,-1)) 
                       for v in isqrtA_p_list] # (v*v).sum(dim=list(range(1,v.dim())))
        E_diag = torch.sum(torch.stack(E_diag_list, dim=0), dim=0)                             
        
        lr = self.low_rank_dimension.item()
        tuple_list = zip(isqrtA_p_list, self._isqrtA_V_cache)
        p_iA_V_list = [torch.einsum('ke,kej->kj', 
                                    tple[0].view(K,-1), 
                                    tple[1].view(K,-1,lr))
                       for tple in tuple_list]
        p_iA_V = torch.sum(torch.stack(p_iA_V_list, dim=0), dim=0) # GMM_comp*lr
        
        E_lr = torch.sum(torch.pow(torch.bmm(self.sqrt_schur_complement_cache, p_iA_V.unsqueeze(2)).squeeze(2), 2), dim=-1)   
        mE = E_lr - E_diag            
        
        #
        log_pi = log_softmax(global_parameters.alpha, dim=0, _stacklevel=5)
        E = -torch.logsumexp(mE*.5 - self._logdet_cache - self._dim_log_sqrt_2pi_cache + log_pi, dim=0) 
                            
        return E 
    
    def variational_parameter_names(self):
        return self.GMMVariationalParameter._fields
    
    def variational_parameter_type(self):
        return self.GMMVariationalParameter
    
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