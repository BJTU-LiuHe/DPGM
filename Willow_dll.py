import numpy as np

from ctypes import *
import numpy.ctypeslib as npct

import scipy.io
import os
import cv2
#import torch as t

import GM_GenData
from GM_GenData import *


WILLOW_CNN_FEATURE_ROOT='data/cnn_features_mat/willow'
WILLOW_LFNET_FEATURE_ROOT='data/lfnet_features/willow'
WILLOW_FILE_ROOT = "data/WILLOW"
WILLOW_TRAIN_NUM = 20
WILLOW_TRAIN_OFFSET = 0

VISFEA_TYPE_RAWPIXEL = GM_GenData.VISFEA_TYPE_RAWPIXEL
VISFEA_TYPE_SIFT = GM_GenData.VISFEA_TYPE_SIFT
VISFEA_TYPE_LFNET = GM_GenData.VISFEA_TYPE_LFNET
VISFEA_TYPE_PRETRAINED_LFNET = GM_GenData.VISFEA_TYPE_PRETRAINED_LFNET
VISFEA_TYPE_PRETRAINED_VGG16 = GM_GenData.VISFEA_TYPE_PRETRAINED_VGG16


def _list_images(root, category) :
    imgFiles = []
    path = os.path.join(root, category)
    for file in os.listdir(path) :
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) :
            if os.path.basename(filepath).endswith('.png') :
                imgFiles.append(filepath)

    return imgFiles

def _load_annotation(file) :
    iLen = len(file)
    raw_file = file[:iLen-4]
    anno_file = raw_file + ".mat"
    anno = scipy.io.loadmat(anno_file)
    pts = np.transpose(anno["pts_coord"])
    return pts

def _load_cnn_feature(file):
    mat_file = WILLOW_CNN_FEATURE_ROOT+file[11:-4]+'.mat'
    infos = scipy.io.loadmat(mat_file)

    pts_fea=infos['pts_features']
    n_pts,dim_feat=pts_fea.shape
    pts_fea=pts_fea.reshape(-1,8)
    pts_fea=np.mean(pts_fea,axis=1)
    pts_fea=pts_fea.reshape((n_pts,-1))
    return pts_fea

def _load_lfnet_feature(file):
    mat_file = WILLOW_LFNET_FEATURE_ROOT+file[11:-4]+'.mat'
    infos = scipy.io.loadmat(mat_file)

    pts_fea=infos['features']
    return pts_fea


def _read_image_features(file, feaType) :
    anno_pts = _load_annotation(file)
    descs = None
    patches = []

    if feaType == VISFEA_TYPE_PRETRAINED_VGG16:
        descs=_load_cnn_feature(file)
        #patches = GM_GenData._compute_image_patches(file, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_PRETRAINED_LFNET:
        descs=_load_lfnet_feature(file)
        #patches = GM_GenData._compute_image_patches(file, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_SIFT:
        kps, descs = GM_GenData._compute_SIFT_features(file, anno_pts)
        #patches = GM_GenData._compute_image_patches(file, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_LFNET:
        kps, descs = GM_GenData._compute_SIFT_features(file, anno_pts)
        patches = GM_GenData._compute_image_patches(file, anno_pts, 16, 16)
    elif feaType == VISFEA_TYPE_RAWPIXEL:
        patches = GM_GenData._compute_image_patches(file, anno_pts, 16, 16)
        descs = np.reshape(patches, (patches.shape[0], -1))
        descs = descs.astype(np.float64)

    return anno_pts, descs, patches


def _gen_affinity_Willow(lib,
                      num_nodes0, tails0, heads0, dists0, angs0,
                      num_nodes1, tails1, heads1, dists1, angs1) :

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

    count = lib.build_affinity_Willow(
        num_nodes0, num_edges0, tails0, heads0, dists0, angs0,
        num_nodes1, num_edges1, tails1, heads1, dists1, angs1,
        gidx1, gidx2, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    return gidx1, gidx2, K

# def _gen_affinity_Willow3(lib,
#                       num_nodes0, tails0, heads0, dists0, angs0,
#                       num_nodes1, tails1, heads1, dists1, angs1,
#                       fea_sim, threshold) :
#
#     num_edges0 = len(tails0)
#     num_edges1 = len(tails1)
#
#     gidx1 = []
#     gidx2 = []
#     num_matches = 0
#     for i2 in range(num_nodes1):
#         for i1 in range(num_nodes0):
#             if fea_sim[i1][i2] >= threshold :
#                 gidx1.append(i1)
#                 gidx2.append(i2)
#                 num_matches = num_matches + 1
#     gidx1 = np.array(gidx1)
#     gidx2 = np.array(gidx2)
#
#
#     tails0 = tails0.astype(np.int)
#     heads0 = heads0.astype(np.int)
#     dists0 = dists0.astype(np.float32)
#     angs0 = angs0.astype(np.float32)
#     tails1 = tails1.astype(np.int)
#     heads1 = heads1.astype(np.int)
#     dists1 = dists1.astype(np.float32)
#     angs1 = angs1.astype(np.float32)
#
#     K = np.zeros(num_matches * num_matches, np.float32)
#
#     count = lib.build_affinity_Willow3(
#         num_nodes0, num_edges0, tails0, heads0, dists0, angs0,
#         num_nodes1, num_edges1, tails1, heads1, dists1, angs1,
#         num_matches, gidx1, gidx2, K)
#     K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))
#
#     return gidx1, gidx2, K


# def _compute_edge_angle(points):
#     num_pts = points.shape[0]
#     x = np.transpose(points)[0]
#     y = np.transpose(points)[1]
#
#     xx = np.tile(x, (num_pts, 1))
#     yy = np.tile(y, (num_pts, 1))
#
#     xdist = xx - np.transpose(xx)
#     ydist = yy - np.transpose(yy)
#
#     ang = np.arctan2(ydist, xdist)
#
#     return ang

# def _gen_affinity_Willow2(lib, points1, descs1, points2, descs2, threshold) :
#
#     distance1 = spatial.distance.cdist(points1, points1)
#     distance2 = spatial.distance.cdist(points2, points2)
#     ang1 = _compute_edge_angle(points1)
#     ang2 = _compute_edge_angle(points2)
#
#     num_nodes1 = points1.shape[0]
#     num_nodes2 = points2.shape[0]
#     num_matches = num_nodes1 * num_nodes2
#
#     distance1 = np.float32(np.reshape(distance1, num_nodes1 * num_nodes1, order = 'C'))
#     distance2 = np.float32(np.reshape(distance2, num_nodes2 * num_nodes2, order='C'))
#     ang1 = np.float32(np.reshape(ang1, num_nodes1 * num_nodes1, order = 'C'))
#     ang2 = np.float32(np.reshape(ang2, num_nodes2 * num_nodes2, order='C'))
#
#     fea_sim = _compute_feature_similarity(descs1, descs2)
#     gidx1 = []
#     gidx2 = []
#     num_matches = 0
#     for i2 in range(num_nodes2):
#         for i1 in range(num_nodes1):
#             if fea_sim[i1][i2] >= threshold :
#                 gidx1.append(i1)
#                 gidx2.append(i2)
#                 num_matches = num_matches + 1
#     gidx1 = np.array(gidx1)
#     gidx2 = np.array(gidx2)
#
#     K = np.zeros(num_matches * num_matches, np.float32)
#     count = lib.build_affinity_Willow2(
#         num_nodes1, num_nodes2,
#         distance1, distance2,
#         ang1, ang2,
#         num_matches,
#         gidx1, gidx2, K)
#     K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))
#
#     return gidx1, gidx2, K



# def _compute_feature_similarity(descs0, descs1):
#     # normalize features
#     norm0 = np.linalg.norm(descs0, axis = 1)
#     norm1 = np.linalg.norm(descs1, axis = 1)
#     for i in range(descs0.shape[0]) :
#         descs0[i][:] = descs0[i][:] / norm0[i]
#     for i in range(descs1.shape[0]) :
#         descs1[i][:] = descs1[i][:] / norm1[i]
#
#     # compute feature similarity via vector cosine
#     fea_sim = np.matmul(descs0, np.transpose(descs1))
#
#     return fea_sim


# def _compute_feature_distance(descs0, descs1) :
#     n0 = descs0.shape[0]
#     n1 = descs1.shape[0]
#     fea_dist = np.zeros((n0, n1))
#     for i0 in range(n0):
#         fea0 = descs0[i0]
#         for i1 in range(n1):
#             fea1 = descs1[i1]
#             dist = fea0 - fea1
#             fea_dist[i0][i1] = np.sqrt(np.sum(dist * dist))
#
#     return fea_dist

def _gen_features_Willow(pts0, tails0, heads0,pts_feas0, patches0, pts1, tails1, heads1,pts_feas1, patches1):
    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    num_edges0 = len(tails0)
    num_edges1 = len(tails1)
    num_matches = num_nodes0 * num_nodes1

    gidx1 = np.zeros(num_matches, np.int)
    gidx2 = np.zeros(num_matches, np.int)
    for i in range(num_matches):
        gidx1[i] = i // num_nodes1
        gidx2[i] = i % num_nodes1

    #node_feaLen = 512
    node_feaLen = pts_feas0.shape[1] + pts_feas1.shape[1]
    edge_feaLen = 8
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)

    for i in range(num_matches):
        cor_node0 = pts_feas0[gidx1[i]]
        cor_node1 = pts_feas1[gidx2[i]]
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

    assignGraph = {"gidx1": gidx1,
                    "gidx2": gidx2,
                    "node_features": node_features,
                    "senders": senders,
                    "receivers": receivers,
                    "edge_features": edge_features,
                    "patches1": patches0,
                    "patches2": patches1}

    return assignGraph

def _gen_random_graph(rand,
                      lib,
                      category,
                      num_outlier_min_max,
                      feaType,
                      use_train_set = True) :
    imgFiles = _list_images(WILLOW_FILE_ROOT, category)
    if use_train_set:
        imgIdx = rand.randint(0, WILLOW_TRAIN_NUM, size=2)
    else:
        imgIdx = rand.randint(WILLOW_TRAIN_NUM, len(imgFiles), size = 2)

    anno_pts0, anno_descs0, patches0 = _read_image_features(imgFiles[imgIdx[0]], feaType = feaType)
    anno_pts1, anno_descs1, patches1 = _read_image_features(imgFiles[imgIdx[1]], feaType = feaType)

    # add outliers to two graphs
    num_outlier = rand.randint(num_outlier_min_max[0], num_outlier_min_max[1])
    if False:#num_outlier > 0
        raise("outlier is not supported up to now!")
        # idx_outlier0 = np.arange(0, sift_pts0.shape[0])
        # idx_outlier1 = np.arange(0, sift_pts1.shape[0])
        # rand.shuffle(idx_outlier0)
        # rand.shuffle(idx_outlier1)
        # idx_outlier0 = idx_outlier0[0:num_outlier]
        # idx_outlier1 = idx_outlier1[0:num_outlier]
        # outlier_pts0 = sift_pts0[idx_outlier0]
        # outlier_pts1 = sift_pts1[idx_outlier1]
        # outlier_descs0 = sift_descs0[idx_outlier0]
        # outlier_descs1 = sift_descs1[idx_outlier1]
        # pts0 = np.vstack((anno_pts0, outlier_pts0))
        # pts1 = np.vstack((anno_pts1, outlier_pts1))
        # descs0 = np.vstack((anno_descs0, outlier_descs0))
        # descs1 = np.vstack((anno_descs1, outlier_descs1))
    else:
        pts0 = anno_pts0
        pts1 = anno_pts1
        descs0 = anno_descs0
        descs1 = anno_descs1

    # randomly re-order
    index0 = np.arange(0, pts0.shape[0])
    rand.shuffle(index0)
    pts0 = pts0[index0]
    descs0 = descs0[index0]
    if patches0 is not None:
        patches0 = patches0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    descs1 = descs1[index1]
    if patches1 is not None:
        patches1 = patches1[index1]


    matchInfo = {"pts1": pts0.copy(),
                 "pts2": pts1.copy()}


    # normalize point coordinates
    pts0 = GM_GenData._normalize_coordinates(pts0)
    pts1 = GM_GenData._normalize_coordinates(pts1)

    # record ground-truth matches
#    gX = np.eye(pts1.shape[0])
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(anno_pts0.shape[0]):
        gX[i][i] = 1.0
    gX = np.transpose(np.transpose(gX[index0])[index1])

    tails0, heads0, dists0, angs0 = GM_GenData._build_delaunay_graph(pts0)
    tails1, heads1, dists1, angs1 = GM_GenData._build_delaunay_graph(pts1)

    assignGraph = _gen_features_Willow(pts0, tails0, heads0, descs0, patches0, pts1, tails1, heads1, descs1, patches1)

    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    aff_gidx1, aff_gidx2, aff_K = _gen_affinity_Willow(lib, num_nodes0, tails0, heads0, dists0, angs0,
                                        num_nodes1, tails1, heads1, dists1, angs1)


    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": category,
             "image1": imgFiles[imgIdx[0]],
             "image2": imgFiles[imgIdx[1]]}

    matchInfo["tails1"] = tails0.copy()
    matchInfo["heads1"] = heads0.copy()
    matchInfo["tails2"] = tails1.copy()
    matchInfo["heads2"] = heads1.copy()
    matchInfo["gidx1"] = gidx1.copy()
    matchInfo["gidx2"] = gidx2.copy()

    affinity = {"gidx1": aff_gidx1,
                "gidx2": aff_gidx2,
                "K": aff_K}

    return assignGraph, image, matchInfo, affinity


def gen_random_graphs_Willow(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max,
                              feaType,
                              use_train_set = True,
                              category_id = -1):

    lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
    lib.build_affinity_Willow.argtypes = [c_int,
                                       c_int,
                                       npct.ndpointer(dtype = np.int, ndim = 1, flags = "C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                       c_int,
                                       c_int,
                                       npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                       npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]


    categories = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

    graphs = []
    images = []
    matchInfos = []
    affinities = []
    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, 5)
        else:
            cid = category_id
        graph, image, matchInfo, affinity  = _gen_random_graph(rand, lib,
                                        categories[cid],
                                        num_outlier_min_max=num_outlier_min_max,
                                        feaType = feaType,
                                        use_train_set = use_train_set)
        graphs.append(graph)
        images.append(image)
        matchInfos.append(matchInfo)
        affinities.append(affinity)

    return graphs, images, matchInfos, affinities


