def build(dataset_name, model_config):
    if dataset_name == "Cifar10":
        from .cifar import cifarTrain
        return cifarTrain(model_config)
    elif dataset_name == "imagenet":
        from .imagenet import imagenetTrain
        return imagenetTrain(model_config)
    elif dataset_name == "WikiText2":
        from .lstm import languageModel 
        return languageModel(model_config)
    elif dataset_name == "cifar100":
        from .cifar100 import cifar100Train
        return cifar100Train(model_config)
    elif dataset_name == "WikiText2_new":
        from .new_lstm import lstmModel
        return lstmModel(model_config)
    elif dataset_name == "svhn":
        from .svhn import svhnTrain
        return svhnTrain(model_config)
    else:
        raise NotImplemented("{} not implemented".format(dataset_name))
