from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from model import VAE
from DataSet import DataPreprocessor
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn.functional as F


def PreK(rank, truth, user_nums, k):
    prek = 0
    ure = 0
    for i in range(user_nums):
        ls = truth.get(i, list())
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


def RecK(rank, truth, user_nums, k):
    reck = 0
    ure = 0
    for i in range(user_nums):
        ls = truth.get(i, list())
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


def NDCGK(rank, truth, user_nums, k):
    ndcgk = 0
    ure = 0
    max_DCGuK = [0] * 6
    for i in range(1, k + 1):
        max_DCGuK[i] = max_DCGuK[i - 1]
        max_DCGuK[i] += 1 / math.log(i + 1, 2)
    for i in range(user_nums):
        ls = truth.get(i, list())
        ts = len(ls)
        if ts == 0:
            continue
        DCGuK = 0
        ure += 1
        for j in range(k):
            if rank[i][j] in ls:
                DCGuK += (np.power(2, 1) - 1) / math.log(j + 2, 2)
        if ts >= 5:
            ndcgk += DCGuK / max_DCGuK[k]
        else:
            ndcgk += DCGuK / max_DCGuK[ts]
    return ndcgk / ure


def MRRK(rank, truth, user_nums, k):
    mrr = 0
    ure = 0
    for i in range(user_nums):
        ls = truth.get(i, list())
        ts = len(ls)
        if ts == 0:
            continue
        ure += 1
        for j in range(k):
            if rank[i][j] in ls:
                mrr += 1 / (j + 1)
                break
    return mrr / ure


def fit(user_nums, item_nums, hidden_dimension, dr_rate, lr, beta_, epochs_, patience, k, dataloader, train_dict
        , vad_dict):
    total_train_step = 0
    total_test_step = 0
    mdl = VAE(item_nums, hidden_dimension, dr_rate)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience, verbose=True)
    for epoch in range(epochs_):
        print(epoch)
        mdl.train()
        total_loss = 0
        for ratting_vac, uid in dataloader:
            out, miu, log_var = mdl(ratting_vac)
            reconstruction_loss = -torch.mean(torch.sum(F.log_softmax(out, 1)*ratting_vac, dim=1))
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + log_var - torch.exp(log_var) - miu ** 2, dim=1))
            loss = reconstruction_loss + beta_ * kl_divergence
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            total_train_step += 1
        if (epoch + 1) % 20 == 0:
            pre, rec, ndcg, mrr = test(mdl, train_dict, vad_dict, user_nums, item_nums, k)
            print("Loss:{}".format(total_loss))
            print("Pre@5:{}".format(pre))
            print("Rec@5:{}".format(rec))
            print("NDCG@5:{}".format(ndcg))
            print("MRR@5:{}".format(mrr))
            writer.add_scalar('\\Train/Loss\\', total_loss, total_train_step)
            writer.add_scalar('\\Test/Pre@5\\', pre, total_test_step)
            writer.add_scalar('\\Test/Rec@5\\', rec, total_test_step)
            writer.add_scalar('\\Test/NDCG@5\\', ndcg, total_test_step)
            writer.add_scalar('\\Test/MRR@5\\', mrr, total_test_step)
            writer.flush()
            total_test_step = total_test_step + 1
        best_ndcg = evaluation(mdl, train_dict, vad_dict, user_nums, item_nums, k)
        print("NDCG@5:{}".format(-best_ndcg))
        early_stopping(best_ndcg, mdl)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break


def evaluation(mdl, train_dict, test_dict, user_nums, item_nums, k):
    mdl.eval()
    with torch.no_grad():
        rank = list()
        for i in range(user_nums):
            exclude = train_dict.get(i, list())
            ratting_vec = torch.tensor([0] * item_nums, dtype=torch.float)
            ratting_vec[exclude] = 1
            out, _, _ = mdl(ratting_vec)
            out = out.numpy()
            out[exclude] = -math.inf
            idx = np.argsort(out)[::-1]
            rank.append(idx)
        ndcg = -NDCGK(rank, test_dict, user_nums, k)
    return ndcg


def test(mdl, train_dict, test_dict, user_nums, item_nums, k):
    mdl.eval()
    with torch.no_grad():
        rank = list()
        for i in tqdm(range(user_nums)):
            exclude = train_dict.get(i, list())
            ratting_vec = torch.tensor([0] * item_nums, dtype=torch.float)
            ratting_vec[exclude] = 1
            out, _, _ = mdl(ratting_vec)
            out = out.numpy()
            out[exclude] = -math.inf
            idx = np.argsort(out)[::-1]
            rank.append(idx)
        pre = PreK(rank, test_dict, user_nums, k)
        rec = RecK(rank, test_dict, user_nums, k)
        ndcg = NDCGK(rank, test_dict, user_nums, k)
        mrr = MRRK(rank, test_dict, user_nums, k)
    return pre, rec, ndcg, mrr


if __name__ == "__main__":
    # Hyper parameters
    beta = 0.2
    d = 200
    drop_rate = 0.5
    epochs = 1000
    batch_size = 100
    learning_rate = 1e-3
    early_stop = 50
    top_k = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "../../../ml-1m/ratings.dat"
    sep = "::"

    # DataLoader
    writer = SummaryWriter('./logs')
    data_preprocessor = DataPreprocessor(path, sep)
    n, m, data_loader, train_set, test_set, vad_set = data_preprocessor.dataloader(batch_size=batch_size, shuffle=True)

    # training
    fit(n, m, d, drop_rate, learning_rate, beta, epochs, early_stop, top_k, data_loader, train_set, vad_set)

    # test
    print("=======test=======")
    model = torch.load('mdl/finish_model.pkl')
    pre_, rec_, ndcg_, mrr_ = test(model, train_set, test_set, n, m, top_k)
    print("Pre@5:{}".format(pre_))
    print("Rec@5:{}".format(rec_))
    print("NDCG@5:{}".format(ndcg_))
    print("MRR@5:{}".format(mrr_))
