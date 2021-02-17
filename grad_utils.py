import torch

def return_grad_norm_per_layer(model_class):
    # model_class.model.eval()
    full_rank_accum = [torch.zeros_like(param, device='cuda:0') for param in
                       model_class.model.parameters()]
    print("Norm full rank accum {}".format([
        torch.norm(l).item() for l in full_rank_accum]))

    print("Norm of gradients before starting in grad norm {}".format([
        torch.norm(l.grad.data).item() for l in model_class.model.parameters()]))
    step_iter = model_class.train_single_iter(for_autoscale=True)
    for grad_train in step_iter:
        for lnum, layer in enumerate(grad_train):
            # full_rank_accum[lnum].data = full_rank_accum[lnum].data + layer.data
            full_rank_accum[lnum].data.add_(layer.data)
        model_class.model.zero_grad()

    print("Norm of gradients after zero grad {}".format([
        torch.norm(l.grad.data).item() for l in model_class.model.parameters()]))

    # import ipdb; ipdb.set_trace()
    norm_list = [torch.norm(l).item() for l in full_rank_accum]
    return (norm_list)
