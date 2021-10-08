# This piece of code is directly inspired from, and copied in part from, pyvarinf
# https://github.com/ctallec/pyvarinf
# The original code and the license for pyvarinf can be found at the link above. A copy of the
# license is included in ./pyvarinf/LICENSE.txt.

# The code below was modified in several ways to accommodate richer variational approximations. 

# pylint: disable=too-many-arguments, too-many-locals
import sys
from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn as nn

from .vp.base import VariationalParameters           
    
    
class Variationalize(nn.Module):
    """ Build a Variational model over the model given as input.
    Changes all parameters of the given model to allow learning of a variational distribution 
        over the parameters using Variational inference and the reparametrization trick.
        
        Arguments:
            model_generator: any routine that can be called as follows to generate a new model based on the desired template. 
                                model = model_generator()                        
                This gives freedom with respect to the initialization of variational parameters (randomization),
                for instance this is potentially useful to break symmetry between variational parameters otherwise playing
                the same role (e.g. several mixture components).
    """
    
    def __init__(self, model_factory, variational_parameters):
        super(Variationalize, self).__init__()       
        VPs = self.variational_parameters = variational_parameters # type of variational parameter distribution
        num_samples = VPs.num_coupled_samples()
        
        self.models = nn.ModuleList([model_factory() for _ in torch.arange(num_samples)])
        self.vp_dict = OrderedDict() # dictionary of variational parameters   
        self._variationalize_module(model_factory)
        self._clean_module_parameters(self.models)
        
        self.vp_list = self._flatten_vp_dict(self.vp_dict['locals']) # this doesn't change, unlike the output of 
                            # flatten_module_parameters, which is sample dependent.        
                
        self.register_buffer('_sample_weights', torch.empty(num_samples))
        self._create_eps_dict() # dictionary of "noise" samples from which to build the parameter samples
    
    def _variationalize_module(self, model_factory):
        self._register_global_parameters()
        
        vp_dict = self.vp_dict['locals'] = OrderedDict()
        self._variationalize_module_parameters(self.models[0], vp_dict, '')
        
        num_passes = self.variational_parameters.num_passes_initialize()
        for i in range(num_passes):
            model_sample = model_factory() 
            self._initialize_variational_parameters(vp_dict, model_sample, i)
        
    def _variationalize_module_parameters(self, module, vp_dict, prefix):        
        # vp names
        vp_names = self.variational_parameters.variational_parameter_names()
        
        # main
        named_params = module._parameters.items()  # pylint: disable=protected-access
        for name, p in named_params:
            if p is not None and p.requires_grad:
                vp = self.variational_parameters.variationalize_parameter(vp_dict, name, p)
                for fld in vp_names:
                    self.register_parameter(prefix + '_' + name + '_' + fld, getattr(vp, fld))

        for mname, sub_module in module.named_children():
            vp_dict[mname] = OrderedDict()
            self._variationalize_module_parameters(sub_module, vp_dict[mname],
                                                   prefix + ('_' if prefix else '') + mname)
    
    @staticmethod
    def _clean_module_parameters(module):
        named_params = module._parameters.items()  # pylint: disable=protected-access
        to_erase = [name for name, p in named_params if p is not None and p.requires_grad]

        for name in to_erase:
            delattr(module, name)

        for mname, sub_module in module.named_children():
            Variationalize._clean_module_parameters(sub_module)
            
    def _initialize_variational_parameters(self, vp_dict, module, i):
        named_params = module._parameters.items()  # pylint: disable=protected-access
        for name, p in named_params:
            if p is not None and p.requires_grad:
                vp = vp_dict[name]
                self.variational_parameters.initialize_variational_parameter(vp, p, i)

        for mname, sub_module in module.named_children():
            self._initialize_variational_parameters(vp_dict[mname], sub_module, i)
                   
    def _register_global_parameters(self):       
        global_parameters = self.variational_parameters.global_parameters()
        if global_parameters is not None:
            self.vp_dict['globals'] = global_parameters
            
            for name, value in global_parameters._asdict().items():
                self.register_parameter('globals_' + name, value)
        else:
            self.vp_dict['globals'] = None          
    
    def _create_eps_dict(self):
        self.eps_dict = OrderedDict()
        self._create_eps_dict_r(self.eps_dict, self.vp_dict)
        
    def _create_eps_dict_r(self, eps_dict, vp_dict):
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):
                eps_dict[name] = OrderedDict()
                self._create_eps_dict_r(eps_dict[name], vp)
            else:
                eps_dict[name] = None                                              
    
    def _sample(self): 
        if self.eps_dict is None:
            self._create_eps_dict()
        
        VPs = self.variational_parameters
        self.eps_dict['globals'] = VPs.sample_globals(self.vp_dict['globals'])
        self._sample_parameter_eps(self.vp_dict['locals'], self.eps_dict['locals'], self.eps_dict['globals'])        
        return
    
    def _sample_parameter_eps(self, vp_dict, eps_dict, global_args):  
        VPs = self.variational_parameters
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):
                self._sample_parameter_eps(vp, eps_dict[name], global_args)
            else:
                eps_dict[name] = VPs.sample_parameter_eps(vp, global_args)
        return
                
    def _rebuild_parameters(self, eps_dict):
        self.variational_parameters.initiate_rebuild_parameters( 
            self.vp_dict['globals'],
            self.vp_list,
            eps_dict['globals'],
            self._flatten_parameter_eps(eps_dict['locals']))
        
        for i, model in enumerate(self.models): 
            self._rebuild_model_parameters(self.vp_dict['locals'], eps_dict['locals'], model, eps_dict['globals'], i)
        self._sample_weights = self.variational_parameters.coupled_sample_weights(self.vp_dict['globals'], eps_dict['globals'],
                                                                                 device=self._sample_weights.device)
                
    def _rebuild_model_parameters(self, vp_dict, eps_dict, module, global_args, i):
        """ Sample from the random part of the variational distribution.
        Build the computational graph corresponding to the computations of the parameters of the given module,
        using the corresponding variational parameters in vp_dict, and the rule used to sample epsilons. 
        If the module has submodules, corresponding subcomputational graphs are also built.
       
        :args dico: a 'tree' dictionnary that contains variational
        parameters for the current module, and subtrees for submodules
        :args module: the module whose parameters are to be rebuilt
        """     
        VPs = self.variational_parameters
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):
                self._rebuild_model_parameters(vp, eps_dict[name], getattr(module, name), global_args, i)
            else:
                setattr(module, name, VPs.rebuild_parameter(vp, eps_dict[name], global_args, i))
    
    def mlog_q(self):
        """
        Computes -log(q(p)) for the current parameter sample(s) p. This depends on the sample.
        """
        
        m_log_q_models = [self.variational_parameters.mlog_q(self.vp_dict['globals'], self.vp_list,
                                                             self._flatten_module_parameters(self.vp_dict['locals'],
                                                                                             model), i)
                          for i,model in enumerate(self.models)]
    
        m_log_q_models = torch.stack(m_log_q_models)
        return torch.dot(m_log_q_models, self._sample_weights.type_as(m_log_q_models)).unsqueeze(0)
        
    def entropy_q(self):
        """
        Computes the entropy <-log q>_q. This doesn't depend on the sample, only on the variational hyperparameters.
        This is useful when splitting -KL[q||p] = <log p>_q - <log q>_q = -<reg()>_q + H(q) + cst.
        The first term can be approximated with a stochastic estimate for generic regularizers; 
            whereas the entropy of q is sometimes available in closed form.
        """
        return self.variational_parameters.entropy_q(self.vp_dict['globals'], self.vp_list)        
    
    @staticmethod
    def _flatten_vp_dict(vp_dict): 
        res = []        
        for key, val in vp_dict.items():
            if isinstance(val, dict):
                res.extend(Variationalize._flatten_vp_dict(val))
            else:
                res.append(val)                

        return res
    
    @staticmethod
    def _flatten_module_parameters(vp_dict, module):
        p_list = []       
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):     
                p_list.extend(Variationalize._flatten_module_parameters(vp, getattr(module, name)))                
            else:
                p_list.append(getattr(module, name))
                    
        return p_list    
    
    @staticmethod
    def _flatten_parameter_eps(eps_dict): 
        eps_list = []
        for name, eps in eps_dict.items():
            if isinstance(eps, dict):     
                eps_list.extend(Variationalize._flatten_parameter_eps(eps))
            else:
                eps_list.append(eps)
                    
        return eps_list 
          
    def forward(self, *inputs, **kwargs):
        """
        A few reasons one might want to fix the sample:
        1. To reuse over multiple iterations
        2. If the optimizer needs to reevaluate the model multiple times (generally via a closure that calls the model)
        3. At test time, to use the same samples across the batch
        
        For 3., we actually provide an option to directy sample and detach a model, using sample_model() instead. 
        The returned model will be entirely detached from the variational model.
        """  
        eps_dict = kwargs.pop('eps', None)
        if eps_dict is None:
            if kwargs.pop('sample', True):
                self._sample()  
                rebuild = True            
            eps_dict = self.eps_dict
            ignore_detach = False
        else:
            ignore_detach = True           
            rebuild = True        
        
        if rebuild or kwargs.pop('rebuild', True):
            self._rebuild_parameters(eps_dict)
        
        result = [model(*inputs) for model in self.models]
        weights = torch.split(self._sample_weights, 1)
        if kwargs.pop('detach_eps', False) and not ignore_detach: 
            self.eps_dict = None
            return result, weights, eps_dict
        return result, weights
    
    def sample_model(self, detach=True):
        self._sample()
        self._rebuild_parameters(self.eps_dict)
        
        if detach:
            # deep copy of models takes some schmilblick, because only tensors that are graph leaves can be deep-copied
            # but "parameters" controlled by underlying vps are n longer actual parameters nor leaves.
            # We detach them, deepcopy, then make them parameters again to normalise the output with normal untampered models.
            with torch.no_grad():
                self._detach_models()
                output_models = copy.deepcopy(self.models)
                self._normalise_detached_models(output_models)
                return output_models, torch.split(self._sample_weights.detach(), 1)
        else:
            return self.models, torch.split(self._sample_weights, 1)
                
    def _detach_models(self):
        """
        Make model attributes that used to be parameters leaves to allow for deepcopy.
        """
        for i, model in enumerate(self.models): 
            self._detach_model_parameters(self.vp_dict['locals'], model)
                
    def _detach_model_parameters(self, vp_dict, module):
        """ 
        Make model attributes that used to be parameters leaves to allow for deepcopy.
        """     
        VPs = self.variational_parameters
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):
                self._detach_model_parameters(vp, getattr(module, name))
            else:
                attr = getattr(module, name).detach()
                setattr(module, name, attr)
                
    def _normalise_detached_models(self, models):
        for i, model in enumerate(models): 
            self._normalise_module(self.vp_dict['locals'], model)
            
    def _normalise_module(self, vp_dict, module):
        VPs = self.variational_parameters
        for name, vp in vp_dict.items():
            if isinstance(vp, dict):
                self._normalise_module(vp, getattr(module, name))
            else:
                attr = getattr(module, name)
                delattr(module, name)
                p = nn.Parameter(attr).requires_grad_(True)
                module.register_parameter(name, p)