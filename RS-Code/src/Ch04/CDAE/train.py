import random

import torch
import pandas as pd
import numpy as np
from model import CDAE
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class GetDataset(Dataset):
    def __init__(self, train_mat):
        self.train_mat = train_mat

    def __getitem__(self, index):
        ratting_vec = torch.tensor(self.train_mat[index], dtype=torch.float)
        uid = torch.tensor([index, ], dtype=torch.long)
        return ratting_vec, uid

    def __len__(self):
        return len(self.train_mat)


def getData(train_data_path, test_data_path, sep):
    cols = ['user_id', 'item_id', 'rating', 'ts']
    base = pd.read_csv(train_data_path, sep=sep, names=cols)
    test = pd.read_csv(test_data_path, sep=sep, names=cols)
    # 创建user-item矩阵
    train_data = base.loc[base['rating'] >= 4].drop('rating', axis=1).reset_index(drop=True)
    test_data = test.loc[test['rating'] >= 4].drop('rating', axis=1).reset_index(drop=True)
    user_list = sorted(list(set(train_data['user_id'].values) | set(test_data['user_id'].values)))
    item_list = sorted(list(set(train_data['item_id'].values) | set(test_data['item_id'].values)))

    # n表示用户数量，m表示物品数量
    uid, iid = len(user_list), len(item_list)

    # 用户id、物品id到矩阵索引的映射
    user_to_index_map = dict(zip(user_list, range(uid)))
    item_to_index_map = dict(zip(item_list, range(iid)))

    def map_rule(x):
        x['user_id'], x['item_id'] = user_to_index_map[x['user_id']], item_to_index_map[x['item_id']]
        return x

    # 映射
    train_data = train_data.apply(map_rule, axis=1)
    test_data = test_data.apply(map_rule, axis=1)
    R = pd.crosstab(index=train_data['user_id'], columns=train_data['item_id'])
    R = R.reindex([i for i in range(uid)], fill_value=0).reindex([i for i in range(iid)], axis=1, fill_value=0)
    R_te = pd.crosstab(index=test_data['user_id'], columns=test_data['item_id'])
    R_te = R_te.reindex([i for i in range(uid)], fill_value=0).reindex([i for i in range(iid)], axis=1, fill_value=0)
    train_mat = np.array(R.values, dtype=np.float32)
    test_mat = np.array(R_te.values, dtype=np.float32)

    P_and_A = R.reset_index().melt(id_vars=['user_id'])
    P_and_A.columns = ['user_id', 'item_id', 'rating']

    # 所有的正样本P和负样本A
    P = P_and_A.loc[P_and_A['rating'] == 1].reset_index(drop=True).drop('rating', axis=1)
    A = P_and_A.loc[P_and_A['rating'] == 0].reset_index(drop=True).drop('rating', axis=1)

    # 所有用户的正样本P_u_set和负样本A_u_set
    P_u_set = P.groupby('user_id').apply(lambda x: list(x['item_id']))
    A_u_set = A.groupby('user_id').apply(lambda x: list(x['item_id']))
    J_u_te_set = test_data.groupby('user_id').apply(lambda x: set(x['item_id'].values)).to_dict()
    return uid, iid, train_mat, test_mat, J_u_te_set, P_u_set, A_u_set


# def get_negative_items(batch_history_data, rate, dr_rate):
#     data = batch_history_data.cpu().numpy()
#     idx = np.zeros_like(data)
#     corrupt_input = np.zeros_like(data)
#     ls = [i for i in range(data.shape[1])]
#     drop_nums = int(dr_rate * data.shape[1])
#     for i in range(data.shape[0]):
#         items = np.where(data[i] == 0)[0].tolist()
#         p_nums = len(np.where(data[i] != 0)[0])
#         items_num = len(items)
#         num = items_num
#         if p_nums * rate < items_num:
#             num = items_num
#         tmp_idx = random.sample(items, num)
#         drop_idx = random.sample(ls, (data.shape[1] - drop_nums))
#         idx[i][tmp_idx] = 1
#         corrupt_input[i][drop_idx] = 1 / (1 - dr_rate) * data[i][drop_idx]
#     idx = torch.tensor(idx).to(device) + batch_history_data
#     corrupt_input = torch.Tensor(corrupt_input).to(device)
#     return idx, corrupt_input


def training(user_nums, item_nums, hidden_dimension, dr_rate, lr, epoch, rate, dataloader, train_mat, test_mat,
             test_dict, p_us, a_us, k):
    total_train_step = 0
    total_test_step = 0
    mdl = CDAE(user_nums, item_nums, hidden_dimension, dr_rate)
    # print(mdl)
    # input()
    # mdl = mdl.to(device)
    loss_fn = nn.LogSigmoid()
    # loss_fn = loss_fn.to(device)
    opt = torch.optim.Adagrad(mdl.parameters(), lr=lr, weight_decay=0.01)
    for e in range(epoch):
        print(e)
        mdl.train()
        total_loss = 0
        for ratting_vec, uid in dataloader:
            # ratting_vec = ratting_vec.to(device)
            u = int(uid)
            # uid = uid.to(device)
            # 用户u的所有正样本和负样本
            P_u, A_u = p_us[u], a_us[u]
            # 负采样
            A_u_sample = random.choices(A_u, k=rate * len(P_u))
            # 正样本和采样负样本的所有物品集合
            i_set = list(set(P_u + A_u_sample))
            out = mdl(uid, ratting_vec)
            # rwave_u = mdl.rwave_u
            ywave_u = torch.where(mdl.rwave_u == 0, -1, 1)
            # loss = -F.logsigmoid(ywave_u[0][i_set]*out[0][i_set]).sum()
            loss = -loss_fn(ywave_u[0][i_set]*out[0][i_set]).sum()
            # loss = F.mse_loss(out[0][i_set], ratting_vec[0][i_set]*rwave_u[0][i_set]).sum()
            # print(out)
            # print(out[i_set])
            # print(ratting_vec[i_set]*rwave_u[i_set])
            # input()

            # loss = loss_fn(out[0][i_set], ratting_vec[0][i_set]*rwave_u[0][i_set])
            # loss = loss_fn(out, ratting_vec * rwave_u)

            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_step = total_train_step + 1
        if (e + 1) % 5 == 0:
            rmse, mse = testing(mdl, train_mat, test_dict, test_mat, k)
            print("Loss:{}".format(total_loss))
            print("Pre@5:{}".format(rmse))
            print("Rec@5:{}".format(mse))
            writer.add_scalar('\\Train/Loss\\', total_loss, total_train_step)
            writer.add_scalar('\\Test/Pre@5\\', rmse, total_test_step)
            writer.add_scalar('\\Test/Rec@5\\', mse, total_test_step)
            writer.flush()
            total_test_step = total_test_step + 1


# 评估函数
def PreK(rank, truth, k):
    prek = 0
    ure = 0
    for i in range(n):
        ls = list(np.where(truth[i] != 0)[0])
        ts = len(ls)
        if ts == 0:
            continue
        preuk = 0
        ure += 1
        for j in range(k):
            if rank[i][j] in ls:
                preuk += 1
        prek += preuk / k
    return prek / ure


def RecK(rank, truth, k):
    reck = 0
    ure = 0
    for i in range(n):
        ls = list(np.where(truth[i] != 0)[0])
        ts = len(ls)
        if ts == 0:
            continue
        recuk = 0
        ure += 1
        for j in range(k):
            if rank[i][j] in ls:
                recuk += 1
        reck += recuk / ts
    return reck / ure


def testing(mdl, train_mat, test_dict, test_mat, k):
    mdl.eval()
    with torch.no_grad():
        users = [i for i in range(train_mat.shape[0])]
        # input_data = torch.tensor(train_mat, dtype=torch.float).to(device)
        # uid = torch.tensor(users, dtype=torch.long).to(device)
        input_data = torch.tensor(train_mat, dtype=torch.float)
        uid = torch.tensor(users, dtype=torch.long)
        out = mdl(uid, input_data)
        out = out.cpu().numpy()
        rank_fism = list()
        for i in range(train_mat.shape[0]):
            exclude = list(np.where(train_matrix[i, :] != 0)[0])
            ls = list()
            for j in range(train_mat.shape[1]):
                if j in exclude:
                    continue
                ls.append((j, out[i, j]))
            ls = sorted(ls, key=lambda x: x[1], reverse=True)
            rank_fism.append([ls[j][0] for j in range(len(ls))])
        pre = PreK(rank_fism, test_mat, k)
        rec = RecK(rank_fism, test_mat, k)
    return pre, rec


if __name__ == '__main__':
    # Hyper parameters
    d = 20
    drop_rate = 0
    q = 0.2
    rho = 5
    epochs = 100
    batch_size = 1
    learning_rate = 0.1
    top_k = 5

    # Data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, m, train_matrix, test_matrix, test_set, P_u_set, A_u_set = getData('../../../ml-100k/u1.base'
                                                                          , '../../../ml-100k/u1.test', sep='\t')
    dataset = GetDataset(train_matrix)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter('./logs')
    # training
    training(n, m, d, drop_rate, learning_rate, epochs, rho, data_loader
             , train_matrix, test_matrix, test_set, P_u_set, A_u_set, top_k)
