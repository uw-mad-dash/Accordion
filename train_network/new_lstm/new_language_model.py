import os
from copy import deepcopy
from typing import Dict, Iterable, List

import spacy
import torch
import torchtext
from spacy.symbols import ORTH
from torch.utils.data import DataLoader

from mean_accumulator import MeanAccumulator
import math
# from ..utils import DistributedSampler
from .model import RNNModel



class Batch:
    def __init__(self, x, y, hidden):
        self.x = x
        self.y = y
        self.hidden = hidden

ITOS = None  # integer to string
STOI = None  # string to integer

class lstmModel(object):
    def __init__ (self, model_config):
        self._device = model_config['device']
        self._batch_size = model_config['batch_size']
        # self._seed = seed
        self._epoch = 0
        self._data_path = model_config['data_path']

        self.text, self.train_loader, self.val_loader, self.full_loader  = define_dataset(
            self._device, "wikitext2", self._data_path, self._batch_size)

        global ITOS
        global STOI

        ITOS = self.text.vocab.itos
        STOI = self.text.vocab.stoi

        self.model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)
        self.state = [param for param in self.model.parameters()]
        self.buffers = [buffers for buffers in self.model.buffers()]
        self.parameter_names = [name for (name, _) in
                                self.model.named_parameters()]
        self._hidden_container = {"hidden": None}

    def train_iterator(self, batch_size):
        self._epoch += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
        self._hidden_container["hidden"] = self.model.init_hidden(batch_size)
        return SplitBatchLoader(
            self.train_loader,
            self._device,
            rank,
            batch_size,
            model=self.model,
            hidden_container=self._hidden_container,
        )

    def full_train_iterator(self, batch_size):
        self._epoch += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
        self._hidden_container["hidden"] = self.model.init_hidden(batch_size)
        return FullSplitBatchLoader(
            self.full_loader,
            self._device,
            rank,
            batch_size,
            model=self.model,
            hidden_container=self._hidden_container,
        )


    def train_single_iter(self, epoch=None, logger=None, for_autoscale=False):
        self.model.train()
        if for_autoscale:
            total_loss = 0
            # there was a bug here we were only using self._batch_size
            full_train_loader = self.full_train_iterator(self._batch_size)
            for i, batch in enumerate(full_train_loader):
                prediction, hidden = self.model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                loss = self._criterion(
                    prediction.view(-1, self.model.ntokens),
                    batch.y.contiguous().view(-1))
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
                grad_array = [param.grad.data for param in
                              self.model.parameters()]
                if i %200  == 0:
                    cur_loss = total_loss / (200)
                    if logger is not None:
                        logger.info('| epoch {:3d} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(epoch, 
                                                                    cur_loss,
                                                                    math.exp(cur_loss)))
                    total_loss = 0
                yield grad_array
        else:
            total_loss = 0
            train_loader = self.train_iterator(self._batch_size)
            for i, batch in enumerate(train_loader):
                prediction, hidden = self.model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                loss = self._criterion(
                    prediction.view(-1, self.model.ntokens),
                    batch.y.contiguous().view(-1))
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
                grad_array = [param.grad.data for param in
                              self.model.parameters()]
                if i %200  == 0:
                    cur_loss = total_loss / (200)
                    if logger is not None:
                        logger.info('| epoch {:3d} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(epoch, 
                                                                    cur_loss,
                                                                    math.exp(cur_loss)))
                    total_loss = 0
                yield grad_array
        self.model.eval()

    # def validate_model(self, logger):
        # rank = torch.distributed.get_rank() if torch.distributed.is_available() else 1
        # self._hidden_container["hidden"] = self.model.init_hidden(
            # self._batch_size)
        # test_loader = FullSplitBatchLoader(
            # self.val_loader,
            # self._device,
            # rank,
            # batch_size=self._batch_size,
            # model = self.model,
            # hidden_container=self._hidden_container
        # )
        # test_model = self.model
        # test_model.eval()
        # total_loss = 0
        # for batch in test_loader:
            # with torch.no_grad():
                # # import ipdb; ipdb.set_trace()
                # prediction, hidden = self.model(batch.x, batch.hidden)
                # self._hidden_container["hidden"] = hidden
                # loss = self._criterion(prediction.view(-1,
                                                       # self.model.ntokens),
                                       # batch.y.contiguous().view(-1))
                # total_loss += loss.item()
        # import ipdb; ipdb.set_trace()
        # logger.info('\nTest set: Average loss: {:.4f}, PPL: {:.4f}\n'.format(
            # total_loss, math.exp(total_loss)))

        # return total_loss
    def validate_model(self, logger):
        rank = torch.distributed.get_rank()
        self._hidden_container["hidden"] = self.model.init_hidden(
            self._batch_size)
        test_loader = SplitBatchLoader(
            self.val_loader,
            self._device,
            rank,
            batch_size=self._batch_size,
            model = self.model,
            hidden_container=self._hidden_container
        )

        test_model = self.model
        test_model.eval()
        mean_metrics = MeanAccumulator()
        for batch in test_loader:
            with torch.no_grad():
                prediction, hidden = self.model(batch.x, batch.hidden)
                self._hidden_container["hidden"] = hidden
                metrics = self.evaluate_prediction(prediction, batch.y)
            mean_metrics.add(metrics)
        mean_metrics.reduce()
        val = mean_metrics.value()
        logger.info('\nTest set: Average loss: {:.4f}, PPL: {:.4f}\n'.format(
            val['cross_entropy'].item(), val['perplexity'].item()))

    def evaluate_prediction(self, model_output, reference):
            """
            Compute a series of scalar loss values for a predicted batch and references
            """
            with torch.no_grad():
                cross_entropy = self._criterion(
                    model_output.view(-1, self.model.ntokens), reference.contiguous().view(-1))
                return {
                    "cross_entropy": cross_entropy.detach(),
                    "perplexity": torch.exp(cross_entropy).detach(),
                }



    def _create_model(self):
        model = define_model(self.text)
        model.to(self._device)
        model.train()
        return model


class SplitBatchLoader(object):
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device, rank, batch_size, model, hidden_container):
        self._dataloader = dataloader
        self._device = device
        self._rank = rank
        self._batch_size = batch_size
        self.model = model
        self._hidden_container = hidden_container

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for i, batch in enumerate(self._dataloader):
            # if i == 0:
            #     print("Data signature", batch.text.view(-1)[0:5].numpy())
            x = batch.text[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
            y = batch.target[:, self._rank * self._batch_size : (self._rank + 1) * self._batch_size]
            hidden = self.model.repackage_hidden(self._hidden_container["hidden"])
            yield Batch(x, y, hidden)

class FullSplitBatchLoader(object):
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device, rank, batch_size, model, hidden_container):
        self._dataloader = dataloader
        self._device = device
        self._rank = rank
        self._batch_size = batch_size
        self.model = model
        self._hidden_container = hidden_container

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for i, batch in enumerate(self._dataloader):
            # if i == 0:
            #     print("Data signature", batch.text.view(-1)[0:5].numpy())
            x = batch.text[:, :]
            y = batch.target[:, :]
            hidden = self.model.repackage_hidden(self._hidden_container["hidden"])
            yield Batch(x, y, hidden)



def define_model(TEXT, rnn_n_hidden=650, rnn_n_layers=3, rnn_tie_weights=True,
                 drop_rate=0.4):
    weight_matrix = TEXT.vocab.vectors
    if weight_matrix is not None:
        n_tokens, emb_size = weight_matrix.size(0), weight_matrix.size(1)
    else:
        n_tokens, emb_size = len(TEXT.vocab), rnn_n_hidden

    model = RNNModel(
        rnn_type="LSTM",
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
    )
    if weight_matrix is not None:
        model.encoder.weight.data.copy_(weight_matrix)
    return model


def define_dataset(
    device,
    dataset_name,
    dataset_path,
    batch_size,
    rnn_use_pretrained_emb=False,
    rnn_n_hidden=650,
    reshuffle_per_epoch=True,
    rnn_bptt_len=30,
):
    # create dataset.
    TEXT, train, valid, test = _get_dataset(dataset_name, dataset_path)

    n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1

    # Build vocb.
    # we can use some precomputed word embeddings,
    # e.g., GloVe vectors with 100, 200, and 300.
    if rnn_use_pretrained_emb:
        try:
            vectors = "glove.6B.{}d".format(rnn_n_hidden)
            vectors_cache = os.path.join(dataset_path, ".vector_cache")
        except:
            vectors, vectors_cache = None, None
    else:
        vectors, vectors_cache = None, None
    TEXT.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache)

    # Partition training data.
    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size * n_workers,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size * n_workers,
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )


    _, full_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=batch_size * 1, # changing n_workers to 1 here
        # as I want the validation code to run on all the workers
        bptt_len=rnn_bptt_len,
        device=device,
        shuffle=reshuffle_per_epoch,
    )

    # get some stat.
    return TEXT, train_loader, val_loader, full_loader

    def get_train_norm(self, saved_path, config):
        # here to save memory and despite previous bad experienes 
        # i am using the same model
        self.model.eval() # shouldn't effect dropout
        total_loss = 0
        full_train_loader = self.full_train_loader(self._batch_size)
        # data loader is full loader
        full_rank_accum = [torch.zeros_like(d) for d in
                           self.model.parameters()]
        for i, batch in enumerate(full_train_loader):
            self.model.zero_grad()
            prediction. hidden = self.model(batch.x, batch.hidden)
            self._hidden_container["hidden"] = hidden
            loss = self._criterion(
                prediction.view(-1, self.model.n_tokens),
                batch.y.contiguous().view(-1))
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
            for idx, mdl in enumerate(self.model.parameters()):
                full_rank_accum[idx].add_(mdl.grad.data)
        norm_val = [torch.norm(lval).item() for lval in full_rank_accum]
        return norm_val





def _get_text():
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok)
    return TEXT

def _get_dataset(name, datasets_path):
    TEXT = _get_text()
    #TODO: Fix the data path story
    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(TEXT, root=datasets_path)
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(TEXT, root=datasets_path)
    return TEXT, train, valid, test
