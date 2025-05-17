# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import glob
import pdb
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class mssl_air(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'person_all_240731'
    # dataset_dir = 'tmm'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(mssl_air, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_air')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> mssl_air loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            # print(img_path,'img_path')
            img_basename = os.path.basename(img_path)
            # print(img_basename,'base')
            pid=int(img_basename.split('_')[0])
            # pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}



        dataset = []
        for img_path in sorted(img_paths):
            # pid, camid = map(int, pattern.search(img_path).groups())
            img_basename = os.path.basename(img_path)
            pid = int(img_basename.split('_')[0])
            camid = int(img_basename.split('_')[4][1:])
            # print(img_basename.split('_')[4][1:],'img_basename.spl')
            # print(img_path,camid,'img_path')
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 2221  or pid ==9999 # pid == 0 means background
            assert 1 <= camid <= 15

            sie_inf = 'double'
            if sie_inf=='double':
                if camid in [1, 2, 3, 6, 7, 8, 11, 12, 13]:
                    height=0
                elif camid in [4, 5, 9, 10, 14, 15]:
                    height = 1
                else:
                    print('Wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            elif sie_inf=='fifteen':
                height = int(img_basename.split('_')[2])
                # print(img_basename, height, 'height')
                height -= 1
            camid -= 1  # index starts from 0

            if relabel: pid = pid2label[pid]
            # if relabel: print(pid,'pid222')

            dataset.append((img_path, self.pid_begin + pid, camid, height))
            # print()
            # dataset.append((img_path, self.pid_begin + pid, camid, 0))
            # print(img_path,'img-path')

        return dataset

# CUDA_VISIBLE_DEVICES=2 python train_clipreid.py --config_file configs/person/vit_clipreid.yml  MODEL.SIE_CAMERA True MODEL.SIE_COE 1.0 MODEL.STRIDE_SIZE '[12, 12]'