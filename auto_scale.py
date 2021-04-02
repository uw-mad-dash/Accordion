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
