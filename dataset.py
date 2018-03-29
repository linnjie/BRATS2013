'''
BRATS2013

train: 30 patients, 4 img type,
test:

'''



import os
import sys
import SimpleITK as sitk
import cv2

import numpy as np

from torch.utils.data import Dataset

from preprocess import ReColor, RandomRotate, SampleVolume, CurriculumWrapper



def LoadOnePersonMha(data_root):
    # data_root is person folder, where VSD *folders* are
    # e.g. BRATS2013/BRATS-2/Image-Data/HG/0001
    folders = os.listdir(data_root)
    person_data = {}
    for folder in folders:
        data_type = folder.split('.')[4] # data_type: MR_Flair, MR_T1, MR_T2, MR_T1c, OT
        filenames = os.listdir(os.path.join(data_root, folder))
        mha_filename = [f for f in filenames if f.split('.')[-1]=='mha']
        mha_file = os.path.join(data_root, folder, *mha_filename)
        mha_data = sitk.ReadImage(mha_file)
        person_data[data_type] = sitk.GetArrayFromImage(mha_data).transpose([1, 2, 0])
    return person_data # {type: data}; goes below

def StackData(person_data):
    stacked_data = []
    label = None # can return None rather than break when loading test data
    for img_type, img_data in sorted(person_data.items()): # sorted by keys
        if img_type == 'OT':
            label = img_data
        else:
            h, w, d = img_data.shape
            img_data.shape = [h, w, d, 1] # add 4th dim to stack different img type
            stacked_data.append(img_data.astype(np.float32))
    stacked_data = np.concatenate(stacked_data, axis=3)
    return stacked_data, label




def DrawLabel(ot_data, max_label):
    color_bar = [
        (0, 0, 0),  # 0 black
        (0, 255, 0),  # 1 green
        (0, 0, 255),  # 2 blue
        (255, 0, 0),  # 3 red
        (255, 255, 255),  # 4 white
        (0, 255, 255),  # 5 cyan
        (255, 255, 0)  # 6 yellow
    ]
    R = np.zeros(ot_data.shape, np.uint8)
    G = np.zeros(ot_data.shape, np.uint8)
    B = np.zeros(ot_data.shape, np.uint8)  # (0, 0, 0) black background
    for label in range(1, max_label + 1):  # start from 1, aka, green mask
        R[ot_data == label] = color_bar[label][0]  # say R[ot_data==1] = color_bar[1][0] R
        G[ot_data == label] = color_bar[label][1]  # G[ot_data==1] = color_bar[1][1] G
        B[ot_data == label] = color_bar[label][2]  # B[ot_data==1] = color_bar[1][2] B as result, #1 green
    return cv2.merge([B, G, R])  # RGB to BGR

def MakeGrid(img_data, cols=8):
    h, w, c = img_data[0].shape
    rows = int(len(img_data)/cols) + (1 if len(img_data) % cols > 0 else 0)
    cols = len(img_data) if rows == 1 else cols

    idx = 0
    concat_img = np.zeros((h*rows, w*cols, c), np.uint8)
    for row_idx in range(rows):
        for col_idx in range(cols):
            if idx >= len(img_data):
                continue
            concat_img[h*row_idx: h*(row_idx+1), w*col_idx: w*(col_idx+1), :] = img_data[idx]
            idx += 1
    return concat_img


def Visualize(person_data):
    # normalize to [0, 255]
    normalized_data = {}
    for img_type, img_data in person_data.items():
        print('Image type: ', img_type, 'Image dim: ', img_data.shape)
        # sample img dim (216, 160, 176) for all img type in train 0001 and 0002
        if img_type == 'OT':
            data = img_data # do not normalize ground truth
        else:
            data = img_data.astype(np.float32)
            data *= 255.0/img_data.max()
        normalized_data[img_type] = data.astype(np.uint8)

    # loop over each time slice (3rd dim)
    for i in range(normalized_data.values()[0].shape[2]):
        print('Frame %d' % i)
        groundtruth = 'OT' in normalized_data.keys() # train/test

        # parse ground truth
        if groundtruth:
            # get one slice
            ot_data = normalized_data['OT'][:, :, i]
            # find contours
            gt = (ot_data > 0).astype(np.uint8)
            _, contours, _ = cv2.findContours(gt.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 1 channel to OpenCV BGR
            ot_data = DrawLabel(ot_data, 4)


        # all imgs *including* OT: get one slice; to 3 channels; save as JPEG
        imgs = {}
        for img_type, img_data in normalized_data.items():
            imgs[img_type] = img_data[:, :, i] # 1 channel

        for img_type, img_data in imgs.items():
            img_data = cv2.merge([img_data, img_data, img_data])

            try:
                os.makedirs('./image/dataset' )
            except:
                pass
            cv2.imwrite('./image/dataset/%03d_%s.jpg' % (i, img_type), img_data)
            imgs[img_type] = img_data # 3 channels

        # draw contours
        if groundtruth and len(contours) > 0:
            for img_type, img_data in imgs.items():
                cv2.drawContours(img_data, contours, -1, (0, 0, 255), 1)

        # save ground truth as JPEG
        if groundtruth and len(contours) > 0:
            cv2.imwrite('./image/dataset/%03d_OT.jpg' % i, ot_data) # overwrite?
            imgs['OT'] = ot_data # overwrite?

        # draw text
        for img_type, img_data in imgs.items():
            cv2.putText(img_data, img_type, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

        # make grid and show
        concat_img = MakeGrid(imgs.values(), 5)
        cv2.imshow('img', concat_img)
        cv2.waitKey(50)


def Normalize(data_list, means, norms):  # what kind of normalization is this?
    ndata_list = []
    if means is not None and norms is not None:  # default param value None
        for i, data in enumerate(data_list):  # data: 4-dim
            data_list[i] = (data - means) / norms
        ndata_list = data_list
    else:  # calculate per patient mean and norm
        h, w, d, c = data_list[0].shape
        denominator = h * w * d
        means = np.zeros((1, 1, 1, c), np.float32)
        norms = np.zeros((1, 1, 1, c), np.float32)

        count = 0
        for data in data_list:
            means += np.sum(data, axis=(0, 1, 2), keepdims=True)
            count += denominator
        means /= count

        for data in data_list:
            ndata_list.append(data - means)

        for data in ndata_list:
            norms += np.sum(np.sqrt(data * data), axis=(0, 1, 2), keepdims=True)
        norms /= count

        for data in ndata_list:
            data /= norms

        print('Mean: ', means, 'Norm', norms)
    return ndata_list, means, norms




class ScanDataset(Dataset): # inherited from torch class Dataset; normalize and transform data here
    def __init__(self, folder_paths, sample_shape=(96, 96, 5), means=None, norms=None, is_train=False):
        self.folder_paths = folder_paths
        self.sample_shape = sample_shape
        self.means = means
        self.norms = norms
        self.is_train = is_train

        data_list, label_list = self.load_data(folder_paths)  # load_data defined in child class
        data_list, means, norms = Normalize(data_list, means, norms)
        self.label_list = label_list
        self.data_list = data_list

        self.set_trans_prob(1.0) # prob to go through transformation (augmentation)?
        self.iter_per_sample = 1

        def __len__(self): # should override to provide size of dataset
            return len(self.data_list) * self.iter_per_sample

        def __getitem__(self, index): # should override; support int indexing from 0 to len(self) exclusive
            index = int(index / self.iter_per_sample) # find floor, if 2, idx becomes 0, 0, 1, 1...
            volume = self.data_list[index] # one person data volume
            label = self.label_list[index]
            if self.is_train: # is_train evaluated here only
                assert(label is not None)
                for trans in self.trans_all: # trans_all defined in set_trans_prob
                    volume, label = trans(volume, label)
                for trans in self.trans_data:
                    volume = trans(volume)
            volume = torch.Tensor(volume.copy()).permute(3, 2, 0, 1) # HWDC to CDHW order??
            if label is not None:
                label = torch.Tensor(label.copy()).permute(2, 0, 1) # HWD to DHW
            return volume, label

        def set_trans_prob(self, prob):
            self.trans_prob = prob
            self.trans_all = [CurriculumWrapper(ReColor(alpha=0.05), prob)] # all means data and label
            self.trans_data = [SampleVolume(dst_shape=self.sample_shape, pos_ratio=0.5),
                               CurriculumWrapper(RandomRotate(random_flip=True),
                               prob)]

        def set_iter_per_sample(self, iter_per_sample):
            self.iter_per_sample = iter_per_sample

        def train(self):
            self.is_train = True # call to set is_train to True

        def eval(self):
            self.is_train = False

class BRATSDataset(ScanDataset): # this class is to load specific dataset and store stats
    def __init__(self, folder_paths, sample_shape=(96, 96, 5), means=None, norms=None, is_train=False):
        self.name = 'BRATS'
        '''differnt, to change'''
        # means = np.array([[[[ 51.95236969,  74.40973663,  81.23361206,  95.90114594]]]], dtype=np.float32)
        # norms = np.array([[[[ 89.12859344,  124.9729538 ,  137.86834717,  154.61538696]]]], dtype=np.float32)
        means = np.array([[[[0.16181767, 0.15569262, 0.15443861, 0.20622088]]]], dtype=np.float32)
        norms = np.array([[[[0.27216652, 0.26292121, 0.25937194, 0.34633893]]]], dtype=np.float32)
        super(BRATSDataset, self).__init__(folders, sample_shape, means, norms, is_train)

    def load_data(self, folder_paths): # folders: e.g. 0001; folder: e.g. VSD...
        data_list = []
        label_list = []
        for folder_path in folder_paths:
            print('Loading %s' % folder)
            person_data = LoadOnePersonMha(folder_path)
            person_data, label = StackData(person_data)
            data_list.append(person_data)
            label_list.append(label)
        print('%d data and %d labels loaded' % (len(data_list), len(label_list)))
        return data_list, label_list






'''
run under BRATS2013
visualize
train root: ./BRATS-2/Image-Data/HG/0001
test root: ./BRATS_Leaderboard/LeaderBoard/HG/0116
'''

if __name__ == '__main__':
    data_root = sys.argv[1]
    person_data = LoadOnePersonMha(data_root)
    Visualize(person_data)