import random
import re
import numpy as np
import torch

import os
from pathlib import Path
import xml.dom.minidom

import GM_GenData
from GM_GenData import *


ROOT_QAPDATA = "/home/6T/lh/data/graph_matching/qapdata/"
ROOT_PROCESSED_DATA = "/home/6T/lh/data/graph_matching/prd_qapdata/"
category = GM_GenData.CLASS_QAPDATA
ROOT_PROCESSED_DATA = os.path.join(ROOT_PROCESSED_DATA, category)

if not os.path.exists(ROOT_PROCESSED_DATA):
    os.makedirs(ROOT_PROCESSED_DATA)

data_list = []
# data_dict = dict()
for file in os.listdir(ROOT_QAPDATA):
    if file.startswith(category) and file.endswith(".dat"):
        dataname = file[:-4]
        data_list.append(dataname)

data_list = sorted(data_list)

def _gen_features_QAP(A0, A1):
    A0_min, A0_max = np.min(A0), np.max(A0)
    A1_min, A1_max = np.min(A1), np.max(A1)
    A0 = (A0 - A0_min) / (A0_max - A0_min)
    A1 = (A1 - A1_min) / (A1_max - A1_min)

    num_nodes0 = A0.shape[0]
    num_nodes1 = A1.shape[0]

    tails0, heads0 = np.nonzero(np.random.random((num_nodes0, num_nodes0)))
    tails1, heads1 = np.nonzero(np.random.random((num_nodes1, num_nodes1)))
    num_edges0 = num_nodes0 ** 2
    num_edges1 = num_nodes1 ** 2
    num_matches = num_nodes0 * num_nodes1

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx1[i] = i / num_nodes1
        gidx2[i] = i % num_nodes1

    node_feaLen = 4
    edge_feaLen = 2
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)

    idx = 0
    for i in range(num_edges0):
        for k in range(num_edges1):

            senders[idx] = tails0[i] * num_nodes1 + tails1[k]
            receivers[idx] = heads0[i] * num_nodes1 + heads1[k]
            edge_features[idx] = np.hstack((A0[tails0[i], heads0[i]], A1[tails1[k], heads1[k]]))
            idx = idx + 1

    assignGraph = {"gidx1": gidx1,
                   "gidx2": gidx2,
                   "node_features": node_features,
                   "senders": senders,
                   "receivers": receivers,
                   "edge_features": edge_features,
                   "patches1": np.zeros(shape=(num_nodes0, 1, 1), dtype=np.uint8),
                   "patches2": np.zeros(shape=(num_nodes1, 1, 1), dtype=np.uint8)}
    return assignGraph


def _gen_affinity_VOC(lib,
                      num_nodes0, tails0, heads0, dists0, angs0,
                      num_nodes1, tails1, heads1, dists1, angs1):
    num_edges0 = len(tails0)
    num_edges1 = len(tails1)
    num_matches = num_nodes0 * num_nodes1

    tails0 = tails0.astype(np.int)
    heads0 = heads0.astype(np.int)
    dists0 = dists0.astype(np.float32)
    angs0 = angs0.astype(np.float32)
    tails1 = tails1.astype(np.int)
    heads1 = heads1.astype(np.int)
    dists1 = dists1.astype(np.float32)
    angs1 = angs1.astype(np.float32)

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    K = np.zeros(num_matches * num_matches, np.float32)

    count = lib.build_affinity_VOC(
        num_nodes0, num_edges0, tails0, heads0, dists0, angs0,
        num_nodes1, num_edges1, tails1, heads1, dists1, angs1,
        gidx1, gidx2, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order='C'))

    return gidx1, gidx2, K

def _kronecker_torch(t1, t2):
    batch_num = t1.shape[0]
    t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
    t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
    if t1.is_sparse and t2.is_sparse:
        tt_idx = torch.stack(t1._indices()[0, :] * t2dim1, t1._indices()[1, :] * t2dim2)
        tt_idx = torch.repeat_interleave(tt_idx, t2._nnz(), dim=1) + t2._indices().repeat(1, t1._nnz())
        tt_val = torch.repeat_interleave(t1._values(), t2._nnz(), dim=1) * t2._values().repeat(1, t1._nnz())
        tt = torch.sparse.FloatTensor(tt_idx, tt_val, torch.Size(t1dim1 * t2dim1, t1dim2 * t2dim2))
    else:
        t1 = t1.reshape(batch_num, -1, 1)
        t2 = t2.reshape(batch_num, 1, -1)
        tt = torch.bmm(t1, t2)
        tt = tt.reshape(batch_num, t1dim1, t1dim2, t2dim1, t2dim2)
        tt = tt.permute([0, 1, 3, 2, 4])
        tt = tt.reshape(batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2)
    return tt

def _gen_random_graph(rand,
                      sample_id):

    name = data_list[sample_id]

    # dat_path = Path(os.path.join(ROOT_QAPDATA, (name + '.dat')))
    # sln_path = Path(os.path.join(ROOT_QAPDATA, (name + '.sln')))
    # dat_file = dat_path.open()
    # sln_file = sln_path.open()
    #
    # def split_line(x):
    #     for _ in re.split(r'[,\s]', x.rstrip('\n')):
    #         if _ == "":
    #             continue
    #         else:
    #             yield int(_)
    #
    # dat_list = [[_ for _ in split_line(line)] for line in dat_file]
    # sln_list = [[_ for _ in split_line(line)] for line in sln_file]
    #
    # prob_size = dat_list[0][0]
    #
    # # read data
    # r = 0
    # c = 0
    # Fi = [[]]
    # Fj = [[]]
    # F = Fi
    # for l in dat_list[1:]:
    #     F[r] += l
    #     c += len(l)
    #     assert c <= prob_size
    #     if c == prob_size:
    #         r += 1
    #         if r < prob_size:
    #             F.append([])
    #             c = 0
    #         else:
    #             F = Fj
    #             r = 0
    #             c = 0
    # Fi = np.array(Fi, dtype=np.float32)
    # Fj = np.array(Fj, dtype=np.float32)
    # assert Fi.shape == Fj.shape == (prob_size, prob_size)
    #
    # # read solution
    # sol = sln_list[0][1]
    # perm_list = []
    # for _ in sln_list[1:]:
    #     perm_list += _
    # assert len(perm_list) == prob_size
    # perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
    # for r, c in enumerate(perm_list):
    #     perm_mat[r, c - 1] = 1
    #
    # Fi_t = torch.from_numpy(np.expand_dims(Fi, axis=0))
    # Fj_t = torch.from_numpy(np.expand_dims(Fj, axis=0))
    #
    # aff_mat = _kronecker_torch(Fj_t, Fi_t)
    # aff_mat = aff_mat.numpy()
    # # perm_mat_t = torch.from_numpy(np.expand_dims(perm_mat, axis=0))
    # # perm_mat_vec = perm_mat_t.transpose(1, 2).contiguous().view(1, -1, 1)
    # # sol_v1 = torch.matmul(torch.matmul(perm_mat_vec.transpose(1, 2), aff_mat), perm_mat_vec).view(-1)

    # adj1, adj2, gX, results = Fi, Fj, perm_mat, sol

    # data_info = data_dict[name]
    # adj1 = data_info["A0"]
    # adj2 = data_info["A1"]
    # gX = data_info["gX"]
    # result = data_info["result"]
    # aff_mat = data_info["aff_mat"]

    assignGraph = torch.load(os.path.join(ROOT_PROCESSED_DATA, name + "_assignGraph.pth"))
    # assignGraph = _gen_features_QAP(adj1, adj2)

    # aff_gidx1, aff_gidx2, aff_K = _gen_affinity_VOC(lib, num_nodes0, tails0, heads0, dists0, angs0,
    #                                     num_nodes1, tails1, heads1, dists1, angs1)

    # gidx1 = assignGraph["gidx1"]
    # gidx2 = assignGraph["gidx2"]
    # solutions = np.zeros(len(gidx1), np.bool)
    # for i in range(len(gidx1)):
    #     if gX[gidx1[i]][gidx2[i]]:
    #         solutions[i] = True
    # assignGraph["solutions"] = solutions

    affinity = None
    # affinity = {"gidx1": aff_gidx1,
    #             "gidx2": aff_gidx2,
    #             "K": aff_K}

    return assignGraph, None, None, affinity



def gen_random_graphs_QAP(rand, batch_size, use_train_set,
                          sample_id):
    lib = None

    graphs = []
    images = []
    matchInfos = []
    affinities = []

    if use_train_set:
        for _ in range(batch_size):
            sid = random.randint(0, len(data_list) -1)
            graph, image, matchInfo, affinity = _gen_random_graph(rand,
                                                                  sid)

            graphs.append(graph)
            images.append(image)
            matchInfos.append(matchInfo)
            affinities.append(affinity)
    else:
        graph, image, matchInfo, affinity = _gen_random_graph(rand,
                                                              sample_id)

        graphs.append(graph)
        images.append(image)
        matchInfos.append(matchInfo)
        affinities.append(affinity)

    return graphs, images, matchInfos, affinities

for dataname in data_list:
    if not os.path.exists(os.path.join(ROOT_PROCESSED_DATA, dataname + "_assignGraph.pth")):
        adj0, adj1, gX, results, aff_mat =  GM_GenData._load_data(ROOT_QAPDATA, dataname)
        sample_dict = {
            "A0": adj0,
            "A1": adj1,
            "gX": gX,
            "result": results,
            "aff_mat": aff_mat
        }

        assignGraph = _gen_features_QAP(adj0, adj1)
        gidx1 = assignGraph["gidx1"]
        gidx2 = assignGraph["gidx2"]
        solutions = np.zeros(len(gidx1), np.bool)
        for i in range(len(gidx1)):
            if gX[gidx1[i]][gidx2[i]]:
                solutions[i] = True
        assignGraph["solutions"] = solutions
        torch.save(assignGraph, os.path.join(ROOT_PROCESSED_DATA, dataname + "_assignGraph.pth"))
        torch.save(sample_dict, os.path.join(ROOT_PROCESSED_DATA, dataname + "_dict.pth"))

        print("assignGraph of " + dataname + " has been saved")
