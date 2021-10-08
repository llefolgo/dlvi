import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor                   

from collections import OrderedDict
from abc import ABC, abstractmethod


class VariationalParameters(ABC, nn.Module):
    
    def __init__(self):
        super(VariationalParameters, self).__init__()
        
    def num_coupled_samples(self):
        """
        For some distributions, inference works much better when using a tuple of joint samples or more.
        In that case, the Variationalize module needs to know in advance that it needs to keep track of more than one model.
        """
        return 1
    
    def coupled_sample_weights(self, global_parameters, global_eps, device=None):
        """
        If outputting several samples at a time, it is sometimes convenient to be able to give them different weights.
        
        For instance if using a GMM for the variational distribution, a strategy is to sample one sample from every mixture
        components, and to weight them according to the variational mixture proportion. This way we can naturally backprop 
        the data matching loss onto the proportion.
        """
        return torch.full((self.num_coupled_samples(),), 1./self.num_coupled_samples(), device=device)
    
    def global_parameters(self):
        """
        Globals for the variational distribution (e.g. mixture proportions of a mixture model).
        If they are parameters they will be registered in the variational model with the name of the coresponding field.
        
        Therefore, either None or a namedtuple is expected.
        """
        return None # can override in children and return a named tuple instead. 
        
    def variationalize_parameter(self, vp_dict, name, p, to_variational=True):
        """
        This generally doesn't need to be overriden. Instead specialize the abstract methods
        _to_variational_parameter and _create_eps
        """
        vp_dict[name] = self._to_variational_parameter(p)
        return vp_dict[name]  
        
    #def create_global_randoms(self):
    #    """
    #    Global eps during the sampling. E.g. the low-rank eps for the low-rank part of the gaussian v-dist.
    #    This is just to create a placeholder if needed (the counterpart of variationalize_parameter).
    #    """
    #    return None # can override in children
    
    @abstractmethod    
    def initialize_variational_parameter(self, vp, p, i, *args):
        pass
    
    @abstractmethod
    def sample_parameter_eps(self, vp, global_args):
        """
        Sample the randomized part of the parameters, cf. reparametrization trick.
        """
        pass
    
    def initiate_rebuild_parameters(self, global_parameters, vp_list, global_eps, eps_list):
        """
        Give us a chance to do something before rebuilding, for instance clearing caches or flagging that we have rebuilt 
        parameters, or actually filling caches.
        """
        return
    
    @abstractmethod
    def rebuild_parameter(self, vp, eps, global_args, i):
        """
        Rebuild the parameter from the variational parameters and the randomized eps, cf. reparametrization trick.
        i goes from 0 to num_coupled_samples()
        """
        pass
            
    def sample_globals(self, global_parameters):
        """
        Can be relevant if we use variational distribution that involve latent variables / hyperpriors.
        E.g. global_parameters can return the hyperprior parameters, vs. this routine would sample the latents over which the
        hyperpriors apply.
        
        Can also sample intermediates used to generate samples here, if they need to be sampled globally rather than at a 
        variational parameter level.
        
        global_parameters as output by global_parameters()
        """       
        return None
    
    def mlog_q(self, global_parameters, vp_list, p_list, i):
        raise NotImplementedError("Subclasses should implement this, if needed.")
    
    def entropy_q(self, global_parameters, vp_list):
        raise NotImplementedError("Subclasses should implement this, if needed.")
    
    @abstractmethod
    def _to_variational_parameter(self, p):
        """
        Takes as input a (template of) parameter p. 
        Outputs an iterable of corresponding variational parameters, that is wrapped into a namedtuple by
        the local_factory when variationalizing the module.
        """        
        pass
    
    #@abstractmethod
    #def _create_eps(self, p):
    #    """
    #    Takes as input a (template of) parameter p. 
    #    Outputs a placeholder tensor, or a set/... of tensors that will hold the random part when sampling parameters 
    #   with the reparametrization trick.
    #    """
    #    pass
    
    @abstractmethod
    def variational_parameter_type(self):
        """
        Return, e.g. GaussianVariationalParameter
        """
        pass
    
    @abstractmethod
    def variational_parameter_names(self):
        """
        Typically, just pass the field names of the namedtuple forming the variational parameter
        """
        pass
    
    def num_passes_initialize(self):
        """
        You can control the number of times _initialize_variational_parameters() will be called in Variationalize class.
        This can be useful as a way to properly randomize e.g. different components in a mixture model variational family.
        """
        
        return 1
    
    @staticmethod
    def _copy_parameter_data(target, source, *args):
        """
        Utility function to copy data from source to target parameter. Useful during the initialization of variational
        parameters.
        """
        
        with torch.no_grad():
            target.copy_(source)
            
    @staticmethod
    def _copy_and_scale_parameter_data(target, source, *args):
        """
        Utility function to copy data from source to target parameter and rescale. 
        Useful during the initialization of variational parameters.
        """
        
        with torch.no_grad():
            target.copy_(source*args[0])
            
    @staticmethod
    def _zero_parameter_data(target, source, *args):
        """
        Utility function for the initialization of variational parameters.
        """
        
        target.data.zero_()
        
    @staticmethod
    def _copy_parameter_data_to_slice(target, source, i, *args):
        with torch.no_grad():
            target.data.select(-1, i).view(source.shape).copy_(source) # view necessary for 1d target