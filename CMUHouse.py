import numpy as np

from scipy import spatial
from scipy.spatial import Delaunay

from ctypes import *
import numpy.ctypeslib as npct

import scipy.io
import math


def _load_data_from_mat(mat_file) :

    data = scipy.io.loadmat(mat_file)  # 读取mat文件
    XTs = data["XTs"]
    num_frames = XTs.shape[0] / 2
    num_points = XTs.shape[1]
    return num_frames, num_points, XTs


# def _build_delaunay_graph(points) :
#
#     A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)
#     distance = spatial.distance.cdist(points, points)
#
#     triangles = Delaunay(points).simplices
#     for tri in triangles:
#         A[tri[0]][tri[1]] = distance[tri[0]][tri[1]]
#         A[tri[0]][tri[2]] = distance[tri[0]][tri[2]]
#         A[tri[1]][tri[2]] = distance[tri[1]][tri[2]]
#         A[tri[1]][tri[0]] = A[tri[0]][tri[1]]
#         A[tri[2]][tri[0]] = A[tri[0]][tri[2]]
#         A[tri[2]][tri[1]] = A[tri[1]][tri[2]]
#
#  #   A = A + np.transpose(A)
#
#     idxs = np.nonzero(A)
#     weights = A[idxs]
#     tails = idxs[0]
#     heads = idxs[1]
#
#     return A, tails, heads, weights

def _build_delaunay_graph(points) :
    A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)
    distance = spatial.distance.cdist(points, points)

    triangles = Delaunay(points).simplices
    for tri in triangles:
        A[tri[0]][tri[1]] = distance[tri[0]][tri[1]]
        A[tri[0]][tri[2]] = distance[tri[0]][tri[2]]
        A[tri[1]][tri[2]] = distance[tri[1]][tri[2]]
        A[tri[1]][tri[0]] = A[tri[0]][tri[1]]
        A[tri[2]][tri[0]] = A[tri[0]][tri[2]]
        A[tri[2]][tri[1]] = A[tri[1]][tri[2]]

    idxs = np.nonzero(A)
    dists = A[idxs]
    tails = idxs[0]
    heads = idxs[1]

    edges = points[heads] - points[tails]
    angs = np.zeros(edges.shape[0], dtype = np.float)
    for i in range(edges.shape[0]):
        angs[i] = math.atan2(edges[i][1], edges[i][0])

    return tails, heads, dists, angs



def _gen_random_affinity(rand,
                         lib,
                         XTs,
                         frame_indexs,
                         num_inner_min_max) :

    max_nodes = 30
#    frame_indexs[1] = frame_indexs[0]

    num_nodes = [rand.randint(num_inner_min_max[0],num_inner_min_max[1]), max_nodes]
#    num_nodes = [10, 10]

    points0 = XTs[frame_indexs[0] * 2 : frame_indexs[0] * 2 + 2][:]
    points1 = XTs[frame_indexs[1] * 2 : frame_indexs[1] * 2 + 2][:]

    points0 = np.transpose(points0)
    points1 = np.transpose(points1)

    # randomly select nodes from frame points
    index0 = np.arange(0, max_nodes, 1)
    index1 = np.arange(0, max_nodes, 1)
    rand.shuffle(index0)
#    rand.shuffle(index1)
    index0 = index0[0:num_nodes[0]]
    index1 = index1[0:num_nodes[1]]
    points0 = points0[index0]
    points1 = points1[index1]

    # record ground-truth matches
    gX = np.eye(XTs.shape[1])
    gX = np.transpose(np.transpose(gX[index0])[index1])

    A0, tails0, heads0, weights0 = _build_delaunay_graph(points0)
    A1, tails1, heads1, weights1 = _build_delaunay_graph(points1)

    num_nodes0 = points0.shape[0]
    num_nodes1 = points1.shape[0]
    # gidx1, gidx2, K = _gen_affinity_CMU(lib,
    #     num_nodes0, tails0, heads0, weights0,
    #     num_nodes1, tails1, heads1, weights1)
    gidx1, gidx2, K = _gen_affinity_CMU2(lib, gX, points0, points1,
                                         num_nodes0, A0,
                                         num_nodes1, A1)

    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True

    # # re-order gidx1
    # rand.shuffle(index0)
    # for i in range(len(gidx1)) :
    #     id = gidx1[i]
    #     gidx1[i] = index0[id]

    return {"K": K,
             "gidx1": gidx1,
             "gidx2": gidx2,
             "solutions": solutions}


def _gen_affinity_CMU2(lib, X, points1, points2,
                       num_nodes1, A1,
                       num_nodes2, A2) :
    distance1 = spatial.distance.cdist(points1, points1)
    distance2 = spatial.distance.cdist(points2, points2)

    gidx1 = []
    gidx2 = []
    num_matches = 0
    for i2 in range(num_nodes2):
        for i1 in range(num_nodes1):
            gidx1.append(i1)
            gidx2.append(i2)
            num_matches = num_matches + 1

    gidx1 = np.array(gidx1)
    gidx2 = np.array(gidx2)
    A1 = np.float32(np.reshape(A1, num_nodes1 * num_nodes1, order = 'C'))
    A2 = np.float32(np.reshape(A2, num_nodes2 * num_nodes2, order = 'C'))
    distance1 = np.float32(np.reshape(distance1, num_nodes1 * num_nodes1, order = 'C'))
    distance2 = np.float32(np.reshape(distance2, num_nodes2 * num_nodes2, order='C'))
    K = np.zeros(num_matches * num_matches, np.float32)

    count = lib.build_affinity_CMUHouse2(
        num_nodes1, num_nodes2, distance1, distance2,
        num_matches, gidx1, gidx2, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    return gidx1, gidx2, K


def _gen_affinity_CMU(lib,
                      num_nodes0, tails0, heads0, weights0,
                      num_nodes1, tails1, heads1, weights1) :


    num_edges0 = len(tails0)
    num_edges1 = len(tails1)
    num_matches = num_nodes0 * num_nodes1

    tails0 = tails0.astype(np.int)
    heads0 = heads0.astype(np.int)
    weights0 = weights0.astype(np.float32)
    tails1 = tails1.astype(np.int)
    heads1 = heads1.astype(np.int)
    weights1 = weights1.astype(np.float32)

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    K = np.zeros(num_matches * num_matches, np.float32)

    count = lib.build_affinity_CMUHouse(
        num_nodes0, num_edges0, tails0, heads0, weights0,
        num_nodes1, num_edges1, tails1, heads1, weights1,
        gidx1, gidx2, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    for i in range(num_matches) :
        K[i][i] = 1.0

    return gidx1, gidx2, K


def _gen_features_CMU(pts0, tails0, heads0, pts1, tails1, heads1):
    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    num_edges0 = len(tails0)
    num_edges1 = len(tails1)
    num_matches = num_nodes0 * num_nodes1

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx1[i] = i / num_nodes1
        gidx2[i] = i % num_nodes1

    node_feaLen = 4
    edge_feaLen = 8
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)

    for i in range(num_matches):
        cor_node0 = pts0[gidx1[i]]
        cor_node1 = pts1[gidx2[i]]
        node_features[i] = np.hstack((cor_node0, cor_node1))

    idx = 0
    for i in range(num_edges0):
        cor_tail0 = pts0[tails0[i]]
        cor_head0 = pts0[heads0[i]]
        for k in range(num_edges1):
            cor_tail1 = pts1[tails1[k]]
            cor_head1 = pts1[heads1[k]]

            senders[idx] = tails0[i] * num_nodes1 + tails1[k]
            receivers[idx] = heads0[i] * num_nodes1 + heads1[k]
            edge_features[idx] = np.hstack((cor_tail0, cor_head0, cor_tail1, cor_head1))

            idx = idx + 1
    # edge_features = np.zeros(shape = (num_edges0 * num_edges1, 1), dtype = np.float)

    assignGraph = {"gidx1": gidx1,
                    "gidx2": gidx2,
                    "node_features": node_features,
                    "senders": senders,
                    "receivers": receivers,
                    "edge_features": edge_features}

    return assignGraph


def _gen_random_graph(rand,
                      XTs,
                      frame_indexs,
                      num_inner_min_max) :

    max_nodes = 30
    num_nodes = [rand.randint(num_inner_min_max[0],num_inner_min_max[1]), max_nodes]

    points0 = XTs[frame_indexs[0] * 2 : frame_indexs[0] * 2 + 2][:]
    points1 = XTs[frame_indexs[1] * 2 : frame_indexs[1] * 2 + 2][:]

    points0 = np.transpose(points0)
    points1 = np.transpose(points1)

    # randomly select nodes from frame points
    index0 = np.arange(0, max_nodes, 1)
    index1 = np.arange(0, max_nodes, 1)
    rand.shuffle(index0)
#    rand.shuffle(index1)
    index0 = index0[0:num_nodes[0]]
    index1 = index1[0:num_nodes[1]]
    points0 = points0[index0]
    points1 = points1[index1]

    # record ground-truth matches
    gX = np.eye(XTs.shape[1])
    gX = np.transpose(np.transpose(gX[index0])[index1])

    tails0, heads0, dists0, angs0 = _build_delaunay_graph(points0)
    tails1, heads1, dists1, angs1 = _build_delaunay_graph(points1)
    assignGraph = _gen_features_CMU(points0, tails0, heads0, points1, tails1, heads1)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    return assignGraph


def gen_random_graphs_CMU(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max):

    mat_file = "cmuHouse.mat"
    num_frames, num_points, XTs =_load_data_from_mat(mat_file)

    # lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
    # lib.build_affinity_CMUHouse.argtypes = [c_int,
    #                                         c_int,
    #                                         npct.ndpointer(dtype = np.int, ndim = 1, flags = "C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                         c_int,
    #                                         c_int,
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]
    #
    # lib.build_affinity_CMUHouse2.argtypes = [c_int,
    #                                         c_int,
    #                                         npct.ndpointer(dtype = np.float32, ndim = 1, flags = "C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                         c_int,
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]


    graphs = []
    max_frames = 111
    for _ in range(num_examples):
        gap = rand.randint(10, 101)
        frame1 = rand.randint(0, max_frames - gap)
        frame_indexs = (frame1, frame1 + gap)
        graph = _gen_random_graph(rand,
                                  XTs,
                                  frame_indexs,
                                  num_inner_min_max=num_inner_min_max)

        graphs.append(graph)

    return graphs

