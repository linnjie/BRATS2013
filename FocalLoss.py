import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






class FocalLoss(nn.Module):
    '''
    Loss(x, class) = - alpha * (1 - softmax(x)[class]) ** gamma * log(softmax(x)[class])

    Args:
        alpha(1D Tensor, Variable): scalar factor, e.g. [a, b, c, d, e] for each class
        gamma(float, double)
        size_average(bool): True by default, losses averaged over each mini-batch.
                            False, losses summed for each mini-batch
    '''

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num

        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):   # one mini_batch
        N = inputs.size(0)  # NCDHW
        C = inputs.size(1)  # class_num
        P = F.softmax(inputs)
        print('inputs.shape: ', inputs.shape, 'targets.shape: ', targets.shape)
        class_mask = inputs.data.new(N, C).fill_(0)  # new: Constructs a new tensor of the *same data type* as self tensor.
        class_mask = Variable(class_mask)  # (N, C)
        ids = targets.view(-1, 1)  # (C, 1)
        class_mask.scatter_(1, ids, 1.)  # scatter_(dim, index, src); class_mask[i][ids[i]] = 1.0
        print('class_mask: ', class_mask)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]  # alpha: (C, 1), ids.data.view(-1): (C,), rearrange
        print('alpha: ', alpha)
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = - alpha * (torch.pow(1 - probs), self.gamma) * log_p

        if self.size_average:
            loss = batch_loss.meam()
        else:
            loss = batch_loss.sum()
        return loss