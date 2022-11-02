import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings

import matplotlib.pyplot as plt


class Rgbt234(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().rgbt234_dir if root is None else root
        super().__init__('RGBT234', root, image_loader)

        # Keep a list of all classes
        # 读出每个类别的文件夹，名称
        self.class_list = [f for f in os.listdir(self.root)]
        # 每个类型的文件夹名称对应一个数字，保存成字典
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        # 得到此数据集包含的类别txt文件序列
        self.sequence_list = self._build_sequence_list(vid_ids, split)

        #
        self.img_name = {}
        for class_name in self.class_list:
            if class_name.split('.')[-1] != 'txt':
                self.img_name[class_name] = [name for name in
                                             os.listdir(os.path.join(self.root, class_name, 'visible'))]
                self.img_name[class_name + 'i'] = [name for name in
                                                   os.listdir(os.path.join(self.root, class_name, 'infrared'))]
                self.img_name[class_name] = sorted(self.img_name[class_name])
                self.img_name[class_name + 'i'] = sorted(self.img_name[class_name + 'i'])

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        # 得到的为seq_per_class['airplane']=[1,2,10,11] seq_per_class['basketball']=[12,13...]
        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        # 读取data_specs中对应的文件txt文本
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'rgbt234_train_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c + '-' + str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        # 例如airplane-1 将其分割为airplane 1 保存到seq_per_class['airplane']=[1,2,3]
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "visible.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values

        bb_anno_file_T = os.path.join(seq_path, "infrared.txt")
        gt_T = pandas.read_csv(bb_anno_file_T, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                               low_memory=False).values

        bb_aano_file_grouth = os.path.join(seq_path, "init.txt")
        gt_grouth = pandas.read_csv(bb_aano_file_grouth, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                                    low_memory=False).values
        return {'visible': torch.tensor(gt), 'infrared': torch.tensor(gt_T), 'grouth': torch.tensor(gt_grouth)}

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = {}
        visible = {}

        valid['visible'] = (bbox['visible'][:, 2] > 0) & (bbox['visible'][:, 3] > 0)
        visible['visible'] = valid['visible'].byte()
        valid['infrared'] = (bbox['infrared'][:, 2] > 0) & (bbox['infrared'][:, 3] > 0)
        visible['infrared'] = valid['infrared'].byte()
        valid['grouth'] = (bbox['grouth'][:, 2] > 0) & (bbox['grouth'][:, 3] > 0)
        visible['grouth'] = valid['grouth'].byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # return os.path.join(seq_path, 'visible', '{:05}'.format(frame_id + 1) + 'v' + '.jpg')  # frames start from 1

        return {'visible': os.path.join(seq_path, 'visible',
                                        self.img_name[seq_path.split('/')[-1]][frame_id]), \
                'infrared': os.path.join(seq_path, 'infrared',
                                         self.img_name[seq_path.split('/')[-1] + 'i'][frame_id].replace('v',
                                                                                                        'i'))}  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        frame = self._get_frame_path(seq_path, frame_id)
        return {'visible': self.image_loader(frame['visible']), 'infrared': self.image_loader(frame['infrared'])}

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        frames = {}
        frames['visible'] = []
        frames['infrared'] = []
        for iv in frame_list:
            frames['visible'].append(iv['visible'])
            frames['infrared'].append(iv['infrared'])
        frame_list = frames
        # plt.imshow(frame_list[0]['visible'])
        # plt.show()
        # plt.imshow(frame_list[0]['infrared'])
        # plt.show()

        if anno is None:
            anno = self.get_sequence_info(seq_id)
            # anno['visible'] = info['visible']
            # anno_T['infrared'] = info['infrared']

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = {}
            anno_frames[key]['visible'] = [value['visible'][f_id, ...].clone() for f_id in frame_ids]
            anno_frames[key]['infrared'] = [value['infrared'][f_id, ...].clone() for f_id in frame_ids]
            anno_frames[key]['grouth'] = [value['grouth'][f_id, ...].clone() for f_id in frame_ids]
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
