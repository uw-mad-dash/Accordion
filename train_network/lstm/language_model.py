import torch
import math

# user libs
from . import data
from . import nlp_model

class languageModel(object):
    """
    Setup NLP model
    """

    def __init__(self, model_config):
        self.device = model_config['device']
        self.corpus = data.Corpus(model_config['data_path'])
        self.train_data = self._batchify(self.corpus.train,
                                         model_config['batch_size'])
        self.val_data = self._batchify(self.corpus.test,
                                       model_config['batch_size'])
        self.ntokens = len(self.corpus.dictionary)
        self.model = nlp_model.RNNModel(model_config['arch'], self.ntokens,
                                        model_config['emsize'],
                                        model_config['nhid'],
                                        model_config['nlayers'],
                                        model_config['dropout'],
                                        model_config['tied']).to(self.device)
        self.model_arch = model_config['arch']
        self.batch_size = model_config['batch_size']
        self.bptt = model_config['bptt']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.lr = model_config['init_lr'] # doing this so can print, otherwise no use
        self.clip = model_config['clip']
    def _batchify(self, data, bsz):
        """
        Some sort of reaarrangement of the data
        Well explained in the original pytorch examples
        """
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)
        
    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

 
    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        if self.model_arch != "Transformer":
            hidden = self.model.init_hidden(self.batch_size)
        self.model.train()
        total_loss = 0
        if for_autoscale:
            for batch_idx, i in enumerate(range(0, self.train_data.size(0) -1,
                                                self.bptt)):
                data, target = self.get_batch(self.train_data, i)
                if self.model == 'Transformer':
                    output = self.model(data)
                else:
                    hidden = repackage_hidden(hidden)
                    output, hidden = self.model(data, hidden)
                loss = self.criterion(output.view(-1, self.ntokens), target)
                loss.backward()
                total_loss += loss.item()
                # NOTE: For curiosity
                # try to see the difference in case of doign this on decoded
                # gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                grad_array = [param.grad.data for param in self.model.parameters()]
                if batch_idx % 200 == 0:
                    cur_loss = total_loss / (200)
                    # break # for speed debugging
                    if logger is not None:
                        # not to log when doing things for auto scale
                        logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch_idx,
                                                                    len(self.train_data)//self.bptt,
                                                                    self.lr, cur_loss,
                                                                    math.exp(cur_loss)))
                        total_loss = 0 
                yield grad_array
        else:
            length_train_set = self.train_data.size(0) - 1
            # TODO: assuming distributed
            local_rank = torch.distributed.get_rank()
            num_workers = torch.distributed.get_world_size()

            start_position = local_rank * int(len(self.train_data)/num_workers)
            end_position = (local_rank+1) * int(len(self.train_data)/num_workers)
            for batch_idx, i in enumerate(range(start_position, end_position-1,
                                                self.bptt)):
                data, target = self.get_batch(self.train_data, i)
                if self.model == 'Transformer':
                    output = self.model(data)
                else:
                    hidden = repackage_hidden(hidden)
                    output, hidden = self.model(data, hidden)
                loss = self.criterion(output.view(-1, self.ntokens), target)
                loss.backward()
                total_loss += loss.item()
                # NOTE: For curiosity
                # try to see the difference in case of doign this on decoded
                # gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                grad_array = [param.grad.data for param in self.model.parameters()]
                if batch_idx % 200 == 0:
                    cur_loss = total_loss / (200)
                    # break # for speed debugging
                    if logger is not None:
                        # not to log when doing things for auto scale
                        logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch_idx,
                                                                    len(self.train_data)//self.bptt,
                                                                    self.lr, cur_loss,
                                                                    math.exp(cur_loss)))
                        total_loss = 0 
                yield grad_array
    def validate_model(self, logger):
        total_loss = 0
        if self.model_arch != "Transformer":
            hidden = self.model.init_hidden(self.batch_size)
        with torch.no_grad():
            for i in range(0, self.val_data.size(0) -1, self.bptt):
                data, target = self.get_batch(self.val_data, i)
                if self.model == 'Transformer':
                    output = self.model(data)
                else:
                    output, hidden = self.model(data, hidden)
                    hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, self.ntokens)
                total_loss += len(data) * self.criterion(output_flat, target).item()
        total_loss /= (len(self.val_data)-1)
        logger.info('\nTest set: Average loss: {:.4f}, PPL: {:.4f}\n'.format(
        total_loss, math.exp(total_loss)))
        return total_loss 

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
