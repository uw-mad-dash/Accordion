import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


class imagenetTrain(object):
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
        model = models.__dict__[arch]()
        model.to(self.device)
        return(model)

    def _create_data_loader(self, data_path, num_workers, train_batch_size,
                            test_batch_size, is_distributed):
        """
        Returns test and train loaders for a given dataset
        
        """
        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        sampler = None
        is_shuffle = True
        if is_distributed:
            sampler = torch.utils.data.DistributedSampler(train_dataset)
            is_shuffle = False
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=is_shuffle,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   sampler=sampler)

        full_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   sampler=None)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])),
            batch_size=test_batch_size, shuffle=False, pin_memory=True)

        return (train_loader, test_loader, full_train_loader)

    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        """
        Train single iter
        """
        self.model.train()
        if not for_autoscale:
            train_data_loader = self.train_loader
        else:
            train_data_loader = self.train_loader
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            grad_array = [param.grad.data for param in self.model.parameters()]
            if batch_idx%20 == 0:
                if logger is not None:
                    # not called by autoscale routine
                    logger.info('Train Epoch(imagenet): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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

