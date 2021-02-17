import torch
from . import cifar_architecture
from torchvision import datasets, transforms
import torch.nn.functional as F

class cifar100Train(object):
    def __init__(self, model_config):
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
        model = getattr(cifar_architecture, arch)()
        model.to(self.device)
        return (model)

    def _create_data_loader(self, data_path, num_workers, train_batch_size,
                            test_batch_size, is_distributed):

        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        transform_train = transforms.Compose([
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15), 
             transforms.ToTensor(), 
             transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([ 
             transforms.ToTensor(), 
             transforms.Normalize(mean, std)])
        training_set = datasets.CIFAR100(root=data_path, train=True,
                                         download=True,
                                         transform=transform_train)
        sampler = None
        is_shuffle = True
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(training_set)
            is_shuffle = False
        train_loader = torch.utils.data.DataLoader(training_set,
                                                   num_workers=num_workers,
                                                   batch_size=train_batch_size,
                                                   sampler = sampler,
                                                   shuffle = is_shuffle,
                                                   pin_memory = True)
        
        full_train_loader = torch.utils.data.DataLoader(training_set,
                                                        num_workers=num_workers,
                                                        batch_size=train_batch_size,
                                                        sampler=None,
                                                        shuffle=False,
                                                        pin_memory=True)

        test_set = datasets.CIFAR100(root=data_path, train=False,
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
        self.model.train()
        if not for_autoscale:
            train_data_loader = self.train_loader
        else:
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
                    logger.info('Train Epoch(cifar100): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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

