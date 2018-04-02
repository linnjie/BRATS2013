import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter


class CollateFn:  # why customize? referred directly in step_one_epoch?
    '''
    merges a list of samples to form a mini-batch
    '''
    def __init__(self):
        pass
    def __call__(self, batch_data):
        volume_list = []
        label_list = []
        for volume, label in batch_data:
            volume_list.append(volume)
            label_list.append(label)
            return torch.stack(volume_list), torch.stack(label_list)

def SegLoss(pred, target, num_classes=5, loss_fn=nn.CrossEntropyLoss()):
    pred = pred.permute(0, 2, 3, 4, 1).contiguous()  # call contiguous() before view(), but why?
    pred = pred.view(-1, num_classes)
    target = target.view(-1)
    loss = loss_fn(pred, target.long())
    return loss

class Solver(object):
    def __init__(self, net, dataset, lr, output_dir):
        self.net = net
        self.dataset = dataset
        self.optimizer = self.create_optimizer(lr)  # what behavior? init lr?
        self.output_dir = output_dir

        self.criterion = None # set outside; lambda p, t
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        self.num_iter = 0  # number of examples passed through; used only in tensorboardX
        self.num_epoch = 0
        self.iter_per_sample = 1  # also in ScanDataset __init__

    def create_optimizer(self, lr):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        return optimizer

    def step_one_epoch(self, batch_size, iter_size=1):
        self.net.cuda()
        self.net.train()  # torch.nn.Module method
        self.dataset.train()
        self.dataset.set_iter_per_sample(self.iter_per_sample)
        batch_data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                                num_workers=batch_size/2, collate_fn=CollateFn(), pin_memory=True)  # pin_memory
        for i_batch, (volume, target) in enumerate(batch_data):
            self.num_iter += batch_size
            volume = Variable(volume).cuda()
            target = Variable(target).cuda()

            # forward
            pred = self.net(volume)
            loss = self.criterion(pred, target)
            self.writer.add_scalar('loss', loss.data[0], self.num_iter)

            # backward
            loss.backward()
            if i_batch % iter_size == 0:  # set iter_size to accumulate gradients and update at one time, more stable
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.writer.file_writer.flush()
        self.num_epoch += self.iter_per_sample
        return loss.data[0]

    def save_model(self):
        model_name = 'epoch_%04d.pt' % (self.num_epoch)
        save_path = os.path.join(self.output_dir, 'model', model_name)
        torch.save(self.net.state_dict(), save_path)
        return save_path



