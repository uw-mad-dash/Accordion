import torch
from . import svhn_architectures

from torchvision import datasets, transforms
import torch.nn.functional as F

class svhnTrain(object):
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
        model = getattr(svhn_architectures, arch)()
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
        sampler = None
        is_shuffle = True
        training_set = datasets.SVHN(
                root=data_path, split='train', download=True, 
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
            ]))
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(training_set)
            # when using sampler you don't use shuffle
            is_shuffle = False

        train_loader = torch.utils.data.DataLoader(training_set,
            num_workers=num_workers,
            batch_size=train_batch_size, sampler=sampler,
            shuffle=is_shuffle, pin_memory=True)

        full_train_loader = torch.utils.data.DataLoader( training_set,
            num_workers=num_workers,
            batch_size=train_batch_size, sampler=None,
            shuffle=False, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_path, split='test', download=True, 
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5)),
            ]),
            ),
            num_workers=num_workers,
            batch_size=test_batch_size, sampler=None,
            shuffle=False, pin_memory=True)
        
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
        logger.info('Test set(svhn): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))
        return test_loss
