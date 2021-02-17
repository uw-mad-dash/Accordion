import torch
import powersgd_grad_original as psgd_original
from timer import Timer
auto_scale_high = 2
auto_scale_low = 1
alpha = 1.1
def metric(*args, **kwargs):
    if True == 0:
        log_metric(*args, **kwargs)
#instead of per layer version use the one we normally
timer = Timer(verbosity_level=2, log_fn=metric)

def run_auto_scale(model_class, candidate_methods, thresh):
    """
    Runs over grad list and returns the right k.
    model_class (training class eg. cifa10) : model class
    candidate_methods (list) : list of candidate methods
    Returns - dictionary for candidate methods
    """

    model_class.model.train()
    full_rank_accum = [ torch.zeros_like(param, device='cuda:0') for param in
                       model_class.model.parameters()]
    # reducer with corresponding 
    psgd_reducer_dict = {k: psgd_original.RankKReducer(42, 'cuda:0', timer,
                                                           rank=k) for k in
                         candidate_methods}
    # grad accumlator
    # will accumlate grad for the specific k
    method_grad_accumlator = {k: [torch.zeros_like(val, device='cuda:0') for
                                  val in full_rank_accum] for k in
                              candidate_methods}

    step_iter = model_class.train_single_iter(for_autoscale=True)
    
    for grad_train in step_iter:
        # got the input grad list
        for lnum, layer in enumerate(grad_train):
            # accumlating gradients
            full_rank_accum[lnum] = full_rank_accum[lnum] + layer
        for k in psgd_reducer_dict:
            # for storing output grad
            grad_out = [torch.zeros_like(val, device='cuda:0') for val in
                        full_rank_accum]
            memories =  [torch.zeros_like(val, device='cuda:0') for val in
                        full_rank_accum]
            _ = psgd_reducer_dict[k].reduce(grad_train, grad_out, memories,
                                            use_memory=True)
            for lnum, layer in enumerate(grad_out):
                method_grad_accumlator[k][lnum] = layer + method_grad_accumlator[k][lnum]
        # zeroing out gradient everytime
        #TODO:Write out the original gradient
        model_class.model.zero_grad()

    model_class.model.eval()
    # calculate norm of each layer
    # NOTE: This is wrong because you have to take the norm of the subtraction
    # instead of subtracting the norm

    # full_rank_accum = [torch.norm(val).item() for val in full_rank_accum]
    # for k in method_grad_accumlator:
        # for idx, val in enumerate(method_grad_accumlator[k]):
            # method_grad_accumlator[k][idx] = torch.norm(val).item()
    
    for k in method_grad_accumlator:
        for idx, val in enumerate(method_grad_accumlator[k]):
            full_grad = full_rank_accum[idx]
            method_grad = val
            if torch.norm(full_grad).item() <0.000000001:
                import ipdb; ipdb.set_trace()
            intermediate_sub = (torch.norm(full_grad-method_grad).item())/(
                torch.norm(full_grad).item())
            method_grad_accumlator[k][idx] = intermediate_sub
            #TODO: Do something to dump method grad_accumlator also
    auto_scale_candidate = list()
    
    candidate_methods_sorted = sorted(candidate_methods)
    for idx in range(len(full_rank_accum)):
        found_method_flag = 0
        for k in candidate_methods_sorted:
            if method_grad_accumlator[k][idx] < thresh:
                # NOTE: I am not happy about doing directly and append 
                auto_scale_candidate.append(k)
                found_method_flag = 1
                break
        if found_method_flag == 0:
            # No method found
            # assing none
            auto_scale_candidate.append(None)
    assert(
        len(auto_scale_candidate) == len(full_rank_accum)),"Length not correct"
    return (method_grad_accumlator, auto_scale_candidate)

# def run_auto_scale_gng(model_class, candidate_method, thresh, prev_norm_list):
    # """
    # Runs autoscale based on insights of gradient norm
    # Candidate Method (int) - The method which if it is trained it will achive
                            # full accuracy

    # Thresh (float) - How much should there be a decrease in the norm of grad
    # from previous epoch
    
    # """
    # # this is to avoid changes in batch norm layers
    # # model_class.model.eval()
    # auto_scale_candidate = list()
    # ratio_list = list()
    # full_rank_accum = [torch.zeros_like(param, device='cuda:0') for param in
                       # model_class.model.parameters()]
    # step_iter = model_class.train_single_iter(for_autoscale=True)
    # for grad_train in step_iter:

        # for lnum, layer in enumerate(grad_train):
            # full_rank_accum[lnum].data = full_rank_accum[lnum].data + layer.data
        # model_class.model.zero_grad()
    # # import ipdb; ipdb.set_trace() 
    # norm_list = [torch.norm(l).item() for l in full_rank_accum]
    # if prev_norm_list is None:
        # # first iteration
        # prev_norm_list = [None] * len(norm_list)
    # for new_norm, prev_norm in zip(norm_list, prev_norm_list):
        # if prev_norm is not None:
            # ratio = (prev_norm - new_norm)/(prev_norm)
            # ratio_list.append(ratio)
            # if ratio < thresh:
                # auto_scale_candidate.append(1)
            # else:
                # auto_scale_candidate.append(4)
        # else:
            # auto_scale_candidate.append(4)
    # return (ratio_list, norm_list, auto_scale_candidate)

def run_auto_scale_gng(current_grad_norms, old_grad_norms, current_epoch):
    """
    current grad norms and old grad norms
    """
    if old_grad_norms is None:
        old_grad_norms = [None]*len(current_grad_norms)
    auto_scale_candidate = []
    ratio_list = []
    for new_norm, prev_norm in zip(current_grad_norms, old_grad_norms):
        if current_epoch !=0:
            if prev_norm != 0.0:
                ratio = (abs(prev_norm - new_norm))/(prev_norm)
                ratio_list.append(ratio)
            else:
                ratio = 9
                ratio_list.append(ratio)
                # if prev norm is zero handle it and give it a high rank
            if ratio < 0.5:
                auto_scale_candidate.append(auto_scale_low)
            else:
                auto_scale_candidate.append(auto_scale_high)
        else:
            auto_scale_candidate.append(auto_scale_high)
    return (auto_scale_candidate, ratio_list)
