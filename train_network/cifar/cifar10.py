import torch
from . import cifar_architectures
from torchvision import datasets, transforms
import torch.nn.functional as F

class cifarTrain(object):
    """
    Setup Cifar training, model config provides all the hyper parameters
    required

    model_config(dict): Dictionary of training config
    """
    def __init__ (self, model_config):
        self.device = model_config['device']
        self.model = self._create_model(model_config['arch'])
        # full train loader doesn't do sampling
        self.train_loader, self.test_loader, self.full_train_loader = self._create_data_loader(
            model_config['data_path'], model_config['num_dataloader_threads'],
        model_config['train_batch_size'], model_config['test_batch_size'],
            model_config['is_distributed'])
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lr = model_config['init_lr']

    def _create_model(self, arch):
        """
        Returns the model skeleton of the specified architecture
        arch(string): Model architecture
        """
        #TODO: Fix this architecture thing
        model = getattr(cifar_architectures, arch)()
        model.to(self.device)
        return (model)

    def _create_data_loader(self, data_path, num_workers, train_batch_size,
                            test_batch_size, is_distributed):
        """
        Returns test and train loaders for a given dataset
        data_path(str): Location of dataset
        num_workers(int): Number of workers for loading data
        train_batch_size(int): Num images in training batch
        test_batch_size(int): Num images in test batch
        """
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            #TODO: Measure the performance effect
                # transforms.ToTensor(),
                # transforms.Lambda(lambda x: F.pad(
                            # Variable(x.unsqueeze(0), requires_grad=False),
                            # (4,4,4,4),mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),

            #TODO: Instead of that seems like we need padding in
            # random crop. It seems like without this data augmentation
            # it will not converge to the the right accuracy
            #TODO: Maybe investigate further
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        # data prep for test set
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        training_set = datasets.CIFAR10(root=data_path, train=True,
                                                    download=True, transform=transform_train)
        sampler = None
        is_shuffle = True
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(training_set)
            # when using sampler you don't use shuffle
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(training_set,
                                                   num_workers=num_workers,
                                                   batch_size=train_batch_size,
                                                   sampler = sampler,
                                                   shuffle=is_shuffle,
                                                   pin_memory=True)

        full_train_loader = torch.utils.data.DataLoader(training_set,
                                                        num_workers=num_workers,
                                                        batch_size=train_batch_size,
                                                        sampler=None,
                                                        shuffle=False,
                                                        pin_memory=True)

        test_set = datasets.CIFAR10(root=data_path, train=False,
                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  num_workers=num_workers,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True)
        return (train_loader, test_loader, full_train_loader)

    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        """
        Train single iter and pack grads in a list and return that list
        """
        if not for_autoscale:
            self.model.train()
            train_data_loader = self.train_loader
        else:
            self.model.eval()
            train_data_loader = self.full_train_loader
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            grad_array = [param.grad.data for param in self.model.parameters()]
            if batch_idx%20 == 0:
                if logger is not None:
                    # not called by autoscale routine
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                    100. * batch_idx / len(self.train_loader), loss.item()))
            yield grad_array

    def validate_model(self, logger):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader)
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        return test_loss

    def get_train_norm(self, saved_path, config):
        local_model = getattr(cifar_architectures, config['arch'])()
        local_model.to("cuda:0")
        saved_model = torch.load(saved_path, map_location="cuda:0")['state']
        local_model.load_state_dict(saved_model)
        # model loaded and ready to go
        # time for data loader
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            #TODO: Measure the performance effect
                # transforms.ToTensor(),
                # transforms.Lambda(lambda x: F.pad(
                            # Variable(x.unsqueeze(0), requires_grad=False),
                            # (4,4,4,4),mode='reflect').data.squeeze()),
                # transforms.ToPILImage(),

            #TODO: Instead of that seems like we need padding in
            # random crop. It seems like without this data augmentation
            # it will not converge to the the right accuracy
            #TODO: Maybe investigate further
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])

        training_set = datasets.CIFAR10(root=config['data_path'], train=True,
                                                    download=True, transform=transform_train)

        full_train_loader = torch.utils.data.DataLoader(training_set,
                                                        num_workers=1,
                                                        batch_size=128,
                                                        sampler=None,
                                                        shuffle=False,
                                                        pin_memory=True)
        # data loader ready
        training_criterion = torch.nn.CrossEntropyLoss().to("cuda:0")
        # loss setup
        full_rank_accum = [torch.zeros_like(d) for d in
                           local_model.parameters()]
        # creating buffers
        local_model.eval()
        # setting eval mode
        for batch_idx, (data,target) in enumerate(full_train_loader):
            local_model.zero_grad() # zeroing gradients
            data, target = data.to("cuda:0"), target.to("cuda:0")
            output = local_model(data)
            loss = training_criterion(output, target)
            loss.backward()
            for idx, mdl in enumerate(local_model.parameters()):
                full_rank_accum[idx].add_(mdl.grad.data)
        norm_val = [torch.norm(lval).item() for lval in full_rank_accum]
        return norm_val
