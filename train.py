import sys
import os
import time

import torch
from torch.autograd import Variable

from solver import Solver, SegLoss
from refine_net import RefineNet
from dataset import BRATSDataset

from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def GetDataset(fold, num_fold, need_train=True, need_val=True): # get train and val set
    data_root = './BRATS-2/Image_Data/HG/'
    HG_folder_paths = [os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root))]
    data_root = './BRATS-2/Image_Data/LG/'
    LG_folder_paths = [os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root))]

    train_folder_paths = [] # till 0001
    val_folder_paths = []
    for i, path in enumerate(HG_folder_paths + LG_folder_paths):
        if i % num_fold == fold:
            val_folder_paths.append(path)
        else:
            train_folder_paths.append(path)

    if need_train and len(train_folder_paths) > 0:
        train_set = BRATSDataset(train_folder_paths, sample_shape=(128, 128, 12), is_train=True) # train_folders: e.g. 0001
    else:
        train_set = None

    if need_val and len(val_folder_paths) > 0:
        val_set = BRATSDataset(val_folder_paths, is_train=False)
    else:
        val_set = None

    return train_set, val_set


def SplitAndForward(net, volume, split_size=31): # x is single volume
    pred = []
    for i, chunk in enumerate(torch.split(volume, split_size, dim=1)):  # CDHW; split tensor into chunks
        result = net(chunk.unsqueeze(0))  # NCDHW: (1, num_classes, num_chunks, H, W )
        pred.append(result.data)   # concat D back
    pred = torch.cat(pred, dim=2)  # NCDHW
    return pred


def Evaluate(net, dataset, data_name):
    net.eval()
    dataset.eval()
    evaluators = [EvalDiceScore(), EvalSensitivity(), EvalPrecision()]

    total_time = 0
    for i in range(len(dataset)):
        start = time.time()
        volume, label = dataset[i]  # simgle volume, label
        print('Processsing %d/%d examples' % (i+1, len(dataset)))
        print('volume.shape: ', volume.shape, 'label.shape: ', label.shape)
        h = volume.shape[2]
        w = volume.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            new_h = (h // 8 + 1) * 8
            new_w = (w // 8 + 1) * 8
            new_volume = torch.zeros(volume.shape[0], volume.shape[1], new_h, new_w)
            new_label = torch.zeros(volume.shape[1], new_h, new_w)
            new_volume[:, :, :h, :w] = volume
            new_label[:, :h, :w] = label
            print('new_volume.shape: ', new_volume.shape, 'new_label.shape: ', new_label.shape)
        else:
            new_volume = volume
            new_label = label
        new_volume = Variable(new_volume).cuda()
        new_label = new_label.cuda()
        pred = SplitAndForward(net, new_volume, 15)  # limit memory usage
        pred = torch.max(pred, dim=1)[1]  # most probable class, (1, D, H, W)
        # max returns (max value, argmax), data type: (Tensor, LongTensor)
        end = time.time()
        print('Time: %f' % (end - start))
        total_time += end - start
        pred = pred.long()

        # 1 necrosis, 2 edema, 3 non-enhancing tumor, 4 enhancing tumor, 0 everything else
        for j in range(5):
            for evaluator in evaluators:
                evaluator.AddResult(pred == i, new_label == i)
    print('Average time: %f' % (total_time/(len(dataset)-1)))

    eval_dict = {}
    for i in range(5):
        for evaluator in evaluators:
            eval_value = evaluator.Eval()
            eval_dict['/'.join([data_name, type(evaluator).__name__, str(i)])] = eval_value
            print('Label %d: %s, %f' % (i, type(evaluator).__name__, eval_value))
    return eval_dict

def Train(train_set, val_set, net, num_epoch, lr, output_dir):
    solver = Solver(net, train_set, 0.0001, output_dir)
    solver.criterion = lambda p, t: SegLoss(p, t, num_classes=5) # pred, target, num_classes set here
    solver.iter_per_sample = 100  # goes to self.dataset.set_iter_per_sample() in Solver
    for i_epoch in range(0, num_epoch, solver.iter_per_sample):  # say, (0, 2000, 100), so i_epoch = 0, 100, 200...
        # train
        solver.dataset.set_trans_prob(i_epoch / 1000.0 + 0.15)
        loss = solver.step_one_epoch(batch_size=10, iter_size=1)
        i_epoch = solver.num_epoch  # num_epoch is accumulated in step_one_epoch by iter_per_sample
        print('Epoch: %d, Loss: %f' % (i_epoch, loss))

        if i_epoch % 100 == 0:
            save_path = solver.save_model()
            print('Save model at %s' % save_path)

        # val
        if i_epoch % 100 == 0:
            eval_dict_val = Evaluate(net, val_set, 'val')
            for key, value in eval_dict_val.items():
                solver.writer.add_scalar(key, value, i_epoch)





if __name__ == '__main__':
    fold = int(sys.argv[1])
    train_set, val_set = GetDataset(fold, num_fold=5)
    print('Size of train set: %d' % len(train_set))
    if val_set is not None:
        print('Size of val set: %d' % len(val_set))

    net = RefineNet(in_channels=4, num_classes=5)

    output_dir = './output/brast_%d' % fold
    try:
        os.makedirs(os.path.join(output_dir, 'model'))
    except:
        pass
    try:
        os.makedirs(os.path.join(output_dir, 'tensorboard'))
    except:
        pass
    Train(train_set, val_set, net, num_epoch=1000, lr=0.0001, output_dir=output_dir)