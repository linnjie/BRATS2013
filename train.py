import sys
import os

from solver import Solver, SegLoss
from refine_net import RefineNet
from dataset import BRATSDataset




def GetDataset(fold, num_fold, need_train=True, need_val=True):
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
        train_set = BRATSDataset(train_folder_paths,sample_shape=(128, 128, 12), is_train=True) # train_folders: e.g. 0001
    else:
        train_set = None

    if need_val and len(val_folder_paths) > 0:
        val_set = BRATSDataset(val_folder_paths, is_train=False)
    else:
        val_sel = None

    return train_set, val_set

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
            save_path = Solver.save_model()
            print('Save model at %s' % save_path)

        '''    LATER!
        # val
        if i_epoch % 100 == 0:
            eval_dict_val = Evaluate(net, val_set, 'val')
            '''


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
    Train(train_set, val_set, net,
        num_epoch=100, lr=0.0001, output_dir=output_dir)