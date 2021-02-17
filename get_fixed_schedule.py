def return_schedule(epoch, num_layers):
    if epoch < 100:
        return [4] * num_layers
    else:
        return [None] * num_layers

