import torch
from timer import Timer
import powersgd_grad
def metric(*args, **kwargs):
    if True == 0:
        log_metric(*args, **kwargs)
timer = Timer(verbosity_level=2, log_fn=metric)

# class psgdSparsify(object):
    # """
    # Instead of running powersgd on the full list this thing runs thing
    # on a single layer
    # """
    # def __init__(self,random_seed, device, n_power_iterations, reuse_query=False, rank=1):
        # self.rank = rank        # k in powersgd
        # self.p_memory = None
        # self.q_memory = None
        # self.reuse_query = reuse_query
        # self.memory_update = None
    # def set_random(self, vector):
        # torch.manual_seed(self.rng.randint(1_000_000_000))
        # vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # # orthogonalize(vector)

    # def return_grad(self, grad_in, grad_out, memory_out, use_memory):
        # """
        # Reduce gradients between the workers in place
        # grad_in (tensor) : Input gradients
        # grad_out (tensor) : Output Gradients
        # memory_out (tensor) : Output memory
        # use_memory (Boolean) : Whether to use memory in the same
        # """
        # #TODO: Need to think about what to do with rank 1 tensors
        # # Should they make it here or somewhere else

        # if use_memory and self.memory_update==None:
            # self.memory_update = torch.zeros_like(grad_in)
        # if use_memory:
            # # add memory term to gradient
            # grad_in = grad_in + mem_term
            # # CAUTION: at this point all references of grad in have changes
            # # be carefull !!
        # memory_is_uninitialized = self.p_memory is None
        # matrix = grad_in.view(grad_in.shape[0], -1)
        # n, m = matrix.shape
        # rank = min(n, m, self.rank)
        # p_size = n * rank  # not sure if we need to flatten the array
        # q_sqize = m * rank
        # if self.p_memory is None:
            # self.p_memory = torch.empty(p_size, device=self.device)
            # self.q_memory = torch.empty(q_size, device=self.device)
        # if self.reuse_query and not memory_is_uninitialized:
            # pass
        # else:
            # self.set_random(q)
        # matrix = grad_in.view(grad_in.shape[0], -1)
        # # tor

        # # I just realized there is no need to rewrite this code at this time
        # # all i need to do is call the old code with a single gradient
        # # will come back to this later

class applySparsify(object):
    """
    Apply sparsify method
    This class will handle everything for now
    Will keep track of everything including method for training
    Applying that method
    """
    def __init__(self, grad_shape, device):
        """
        Begining everything is empty no need to initialize anything
        """
        self.k = None  # None also indicates to use Full Ranks
        self.psg_instance = None # Class instance of powerSGD
        # this is for the memory of powerSGD
        #TODO: Currently memory is being dealt in the powersgd method
        # don't do that
        self.device = device
        self.memory  = [torch.zeros(grad_shape, device=self.device)]
        self.distributed = torch.distributed.is_available()
        if self.distributed:
            self.n_workers = torch.distributed.get_world_size()
        else:
            self.n_workers = 1
        # TODO: Add random seed as an argument


    def apply_method(self,grad_in, use_memory=True):
        """
        Applies whatever method is stored in the psg instance
        Use memory is currently not used
        """
        grad_out = [torch.zeros_like(grad_in, device=self.device)]
        grad_in = grad_in + self.memory[0]
        #TODO: Verify memory is being applied properly
        #TODO: Add distributed in the case of num_workers > 1
        if self.psg_instance is not None:
            floats_comm = self.psg_instance.reduce([grad_in], grad_out, self.memory)
            return (grad_out[0], floats_comm)
        else:
            # This is essentially full rank
            if self.distributed:
                # distributed for full rank sgd
                floats_comm = torch.numel(grad_in)
                torch.distributed.all_reduce(grad_in, async_op=False)
                grad_in[:] /=self.n_workers
            return (grad_in, floats_comm)

    def update_method(self, new_k, zero_memory=False):
        if new_k == None:
            self.psg_instance = None
            # didn't find a suitable method
            # added for this case
            self.k = new_k
            self.memory = [torch.zeros_like(self.memory[0],
                                            device=self.device)]

        # making a hack to test what happens when we switch
        elif self.k != new_k:
            # essentially every update call will initialize a new method
            self.psg_instance = powersgd_grad.RankKReducer(42, 'cuda:0', timer,
                                                           rank=new_k)
            self.k = new_k
            if zero_memory:
                self.memory = [torch.zeros_like(self.memory[0],
                                                device=self.device)]

        # elif self.k != new_k:
            # #TODO: Make device configurable
            # self.psg_instance = powersgd_grad.RankKReducer(42, 'cuda:0', timer,
                                                           # rank=new_k)
            # self.k = new_k
        # elif self.k == new_k:
            # # Do nothing
            # pass
        else:
            print ("Do nothing self.k ={}  new_k = {}".format(self.k, new_k))
        return None



