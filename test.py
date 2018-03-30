import os
import sys
import threading

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import SimpleITK as sitk
import cv2
import numpy as np

from refine_net import RefineNet
from dataset import BRATSDataset, DrawLabel, FindMhaFilename
from train import SplitAndForward, GetDataset, Resize

def GetID(folder_path):
    folders = os.listdir(folder_path)
    for folder in folders:
        filename = FindMhaFilename(folder_path, folder)
        print('filename:', filename)
        id = filename.split('.')[-2]
        print('id: ', id)
        return id

def Cvt2Mha(pred):
    pred = pred.astype(np.uint8)
    mha_data = sitk.GetImageFromArray(pred)
    return mha_data


def PredictWorker(net, volume, cuda_id, result, lock):
    print(cuda_id)
    net.eval()
    net.cuda(cuda_id)
    volume = volume.cuda(cuda_id)
    pred = SplitAndForward(net, volume, split_size=15)  # output NCDHW
    pred = F.softmax(Variable(pred.squeeze()), dim=0).data  # squeeze out N; softmax?
    with lock:
        result[cuda_id] = pred


def Evaluate(nets, dataset, output_dir=None):
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder_path = dataset.folder_paths[i]
        print('Processing %d %s' % (i, folder_path))
        volume = Resize(volume)
        volume, _ = Variable(volume, volatile=True)  # inference mode (test time)
        lock = threading.Lock()  # ?
        result = {}
        if len(nets) > 1:
            threads = [threading.Thread(target=PredictWorker,
                                        args=(net, volume, cuda_id, result, lock))
                       for cuda_id, net in enumerate(nets)]  # one net per thread (cuda_id)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            pred = result[0]
            for j in range(1, len(nets)):
                pred += result[j].cuda(0)
        else:
            PredictWorker(nets[0], volume, 0, result, lock)
            pred = result[0]
        pred = torch.max(pred, dim=0)[1]
        pred = pred.cpu().numpy()
        print('pred.shape: ', pred.shape)

        # save result
        if output_dir is not None:
            mha_data = Cvt2Mha(pred)
            mha_file = 'VSD.%s.%s.mha' % (folder_path.split('/')[-1], GetID(folder_path))
            sitk.WriteImage(mha_data, os.path.join(output_dir, mha_file))
        else:
            for j in range(pred.shape[0]):  # i: one volume, j: one slice
                pred = DrawLabel(label_data=pred[i, :, :], max_label=4)
                cv2.imwrite('/image/test/%03d_pred.jpg' % i, pred)
                cv2.imshow('pred', pred)
                cv2.waitKey(50)


def GetTestSet(mode):
    if mode == 'test':
        data_root = './BRATS_Leaderboard/LeaderBoard/HG/'
        HG_folder_paths = [os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root))]
        data_root = './BRATS_Leaderboard/LeaderBoard/LG/'
        LG_folder_paths = [os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root))]

        test_folder_paths = HG_folder_paths + LG_folder_paths  # path ends with 0116, etc
        test_set = BRATSDataset(test_folder_paths, is_train=False)
    elif mode.isdigit():  # to test if a **str** is digit
        fold = int(mode)
        _, test_set = GetDataset(fold, num_fold=5, need_train=False, need_val=True)
    else:
        test_folder_path = mode
        test_set = BRATSDataset([test_folder_path], is_train=False)

    return test_set

def GetModel(model_file_path):
    net = RefineNet(4, 5)
    state_dict = torch.load(model_file_path)
    net.load_state_dict(state_dict)
    return net


if __name__ == '__main__':
    model_file_paths = sys.argv[1]
    mode = sys.argv[2]  # 'test': get all test result
                        # int: get result of one train fold (for validation)
                        # single test_folder_path: speed test

    nets = [GetModel(path) for path in model_file_paths.split(',')]
    print('Nets loaded.')

    test_set = GetTestSet(mode)
    print('Test set loaded.')

    output_dir = os.path.join('./result_BRATS')
    try:
        os.makedirs(output_dir)
    except:
        pass

    Evaluate(nets, test_set, output_dir)