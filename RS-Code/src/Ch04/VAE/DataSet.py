import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class GetDataset(Dataset):
    def __init__(self, train_set, item_nums):
        self.train_set = train_set
        self.item_nums = item_nums

    def __getitem__(self, index):
        ratting_vec = torch.tensor([0]*self.item_nums, dtype=torch.float)
        ratting_vec[self.train_set.get(index)] = 1
        uid = torch.tensor([index, ], dtype=torch.long)
        return ratting_vec, uid

    def __len__(self):
        return len(self.train_set)


class DataPreprocessor:
    def __init__(self, path, sep):
        self.path = path
        self.sep = sep

    def getData(self):
        cols = ['user_id', 'item_id', 'rating', 'ts']
        base = pd.read_csv(self.path, sep=self.sep, names=cols)
        base = base.loc[base['rating'] >= 1].drop(['rating', 'ts'], axis=1).reset_index(drop=True)
        # 用户id、物品id到矩阵索引的重排映射
        user_list = sorted(list(set(base.user_id)))
        item_list = sorted(list(set(base.item_id)))
        uid, iid = len(user_list), len(item_list)
        user_to_index_map = dict(zip(user_list, range(uid)))
        item_to_index_map = dict(zip(item_list, range(iid)))
        # 划分训练（60%）、验证（20%）、测试（20%） 数据集
        idx = list(base.index.values)
        total_num = len(idx)
        vad_num = int(total_num * 0.2)
        te_num = int(total_num * 0.2)
        tmp_idx = random.sample(idx, k=vad_num)
        vad_data = base.iloc[tmp_idx, :].reset_index(drop=True)
        tmp = base.drop(tmp_idx, axis=0).reset_index(drop=True)
        idx = list(tmp.index.values)
        tmp_idx = random.sample(idx, k=te_num)
        test_data = tmp.iloc[tmp_idx, :].reset_index(drop=True)
        train_data = tmp.drop(tmp_idx, axis=0).reset_index(drop=True)
        # 创建user-item 集合
        train_set = dict()
        test_set = dict()
        vad_set = dict()
        for i in train_data.itertuples():
            train_set.setdefault(user_to_index_map.get(i[1]), list()).append(item_to_index_map.get(i[2]))
        for i in test_data.itertuples():
            test_set.setdefault(user_to_index_map.get(i[1]), list()).append(item_to_index_map.get(i[2]))
        for i in vad_data.itertuples():
            vad_set.setdefault(user_to_index_map.get(i[1]), list()).append(item_to_index_map.get(i[2]))
        return uid, iid, train_set, test_set, vad_set

    def dataloader(self, batch_size, shuffle):
        uid, iid, train_set, test_set, vad_set = self.getData()
        dataset = GetDataset(train_set, iid)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return uid, iid, data_loader, train_set, test_set, vad_set

