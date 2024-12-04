import numpy as np
import torch
from scipy import spatial
from scipy.spatial import Delaunay

from ctypes import *
import numpy.ctypeslib as npct
import random
import scipy.io
import os
import cv2
import xlrd
import  xml.dom.minidom

import Siftor

import pascal_voc

import GM_GenData
from GM_GenData import *

import xml.etree.ElementTree as ET

ROOT = "/home/6T/lh/data/graph_matching/"

VOC_CNN_FEA_ROOT=ROOT + 'cnn_features_mat/PascalVoc'
VOC_LFNET_FEA_ROOT=ROOT + 'lfnet_features/PascalVoc'
VOC_ANNOTATION_ROOT = ROOT + "PascalVoc/annotations"
VOC_FEAUTURES_ROOT = ROOT + "PascalVoc/annotations_features"
VOC_BBGM_FEATURE_ROOT = ROOT + "BBGM_features/PascalVoc"
VOC_IMAGE_ROOT = ROOT + "PascalVoc/JPEGImages"


VOC_CATEGORIES = ["aeroplane",   "bicycle", "bird",  "boat",      "bottle", "bus",         "car",   "cat",   "chair", "cow",
                  "diningtable", "dog",     "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",  "train", "tvmonitor"]
VOC_ANNO_LISTS= []

adj_file = "learned_adj_train.xlsx"   #learned_adj_train.xlsx   semantic_adjacency
book = xlrd.open_workbook(adj_file)
catergories = book.sheet_names()
semantic_adj_dict = dict()
pre_semantic_adj_dict = dict()


for catergory in catergories:
    sheet = book.sheet_by_name(catergory)
    rows = sheet.nrows
    cols = sheet.ncols

    kptNames = sheet.row_values(0)[1:]
    semanticAdj = np.zeros((rows - 1, cols - 1))
    for row_idx in range(1, rows):
        for col_idx in range(1, cols):
            semanticAdj[row_idx - 1, col_idx - 1] = sheet.cell_value(row_idx, col_idx)

    if "learned" in adj_file:
        semanticAdj = (semanticAdj + semanticAdj.transpose((1,0)))*0.5
        pre_semanticAdj = semanticAdj
        for idx in range(semanticAdj.shape[0]):
                semanticAdj[idx,idx]=0


        percent = np.percentile(semanticAdj,70)
        semanticAdj[semanticAdj >= percent] = 1
        semanticAdj[semanticAdj < percent] =0
    semantic_adj_dict[catergory] = {"kptNames": kptNames, "semanticAdj": semanticAdj}
    pre_semantic_adj_dict[catergory] = {"kptNames": kptNames, "semanticAdj": pre_semanticAdj}


def semantic_adj(ktpNames, category):
    # print("category",category)

    # print("kptNames:",kptNames)
    # print("semantic_adj_dict[category]",semantic_adj_dict[category]["kptNames"])

    index_adj = [semantic_adj_dict[category]["kptNames"].index(name) for name in ktpNames]
    matrix_adj = semantic_adj_dict[category]["semanticAdj"][index_adj, :][:, index_adj]
    A = matrix_adj
    # matrix_adj = matrix_adj + matrix_adj.transpose((1,0))
    # A = np.zeros_like(matrix_adj)
    # index_sorted = np.argsort(matrix_adj, axis=1)[:, -3:]
    # for idx_row in range(index_sorted.shape[0]):
    #     for idx_col in index_sorted[idx_row]:
    #         A[idx_row, idx_col] = 1

    row_nonzero, col_nonzero = np.nonzero(A)
    return row_nonzero, col_nonzero

def _list_annotations(root, category) :
    xmlFiles = []
    path = os.path.join(root, category)
    for file in os.listdir(path) :
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) :
            if os.path.basename(filepath).endswith('xml'):
                xmlFiles.append(filepath)
    return xmlFiles

def _calc_rate_indexs(rates):
    sum_rates = np.zeros(shape = rates.shape, dtype = rates.dtype)
    sum_rates[0] = rates[0]
    for i in range(1, len(rates)):
        sum_rates[i] = sum_rates[i-1] + rates[i]

    rate_index = np.zeros(sum_rates[-1], np.int)
    index = 0
    for i in range(sum_rates[-1]):
        if i >= sum_rates[index]:
            index = index + 1
        rate_index[i] = index

    return rate_index

def _compute_rate_indexs():
    num_rates = []
    for i in range(len(VOC_CATEGORIES)):
        annoList = _list_annotations(VOC_ANNOTATION_ROOT, VOC_CATEGORIES[i])
        num_rates.append(len(annoList))
    num_rates = np.array(num_rates)
    sum_rates = np.sum(num_rates)
    rate_indexs = _calc_rate_indexs(num_rates)
    return sum_rates, rate_indexs

def _get_bound(file):
    tree = ET.parse(file)
    root = tree.getroot()

    bounds = root.find('./visible_bounds').attrib

    xmin = float(bounds['xmin'])
    ymin = float(bounds['ymin'])
    h = float(bounds['height'])
    w = float(bounds['width'])
    xmax = float(xmin) + float(w)
    ymax = float(ymin) + float(h)

    return xmin, ymin, h, w

def _load_annotation(file):
    dom = xml.dom.minidom.parse(file)
    root = dom.documentElement

    image = root.getElementsByTagName('image')[0].firstChild.data

    keypoints = root.getElementsByTagName('keypoints')[0]
    kps = keypoints.getElementsByTagName('keypoint')

    xmin, ymin, h, w = _get_bound(file)

    annoName = []
    annoPts = []
    for kp in kps:
        x = (float(kp.getAttribute('x')) - xmin) * 256 / w
        y = (float(kp.getAttribute('y')) - ymin) * 256 / h
        name = kp.getAttribute('name')
        annoName.append(name)
        annoPts.append([x, y])

    annoName = np.array(annoName)
    annoPts = np.array(annoPts)

    return image, annoName, annoPts

def _load_cnn_feature(file):
    mat_file = VOC_CNN_FEA_ROOT+file[26:-4]+'.mat'
    infos = scipy.io.loadmat(mat_file)

    pts_fea=infos['pts_features']
    n_pts, dim_feat = pts_fea.shape
    if n_pts==0:
        return None
    pts_fea = pts_fea.reshape(-1, int(MEAN_POOLING_INTERVAL_PASCAL))
    pts_fea = np.mean(pts_fea, axis=1)
    pts_fea = pts_fea.reshape((n_pts, -1))
    return pts_fea

def _load_bbgm_feature(file):
    mat_file = VOC_BBGM_FEATURE_ROOT+file.split("annotations")[-1][:-4]+'.mat'
    infos = scipy.io.loadmat(mat_file)

    pts_fea=infos['pts_features']
    n_pts, dim_feat = pts_fea.shape
    if n_pts==0:
        return None
    # pts_fea = pts_fea.reshape(-1, int(MEAN_POOLING_INTERVAL_PASCAL))
    # pts_fea = np.mean(pts_fea, axis=1)
    # pts_fea = pts_fea.reshape((n_pts, -1))
    return pts_fea

def _mean_pooling(pts_fea):

    n_pts, dim_feat = pts_fea.shape
    pts_fea = pts_fea.reshape(-1, int(MEAN_POOLING_INTERVAL_PASCAL))
    pts_fea = np.mean(pts_fea, axis=1)
    pts_fea = pts_fea.reshape((n_pts, -1))

    return pts_fea

def create_mask(shape, probability):
    prob_value = probability * 1000
    mask = np.random.randint(1, 1000, size= shape)*1.0
    mask[mask < prob_value] = 0
    mask[mask >= prob_value] = 1

    return mask

def _load_lfnet_feature(file):
    mat_file = VOC_LFNET_FEA_ROOT+file[26:-4]+'.mat'
    infos = scipy.io.loadmat(mat_file)

    pts_fea=infos['features']

    return pts_fea


def _compute_keypoint_features(annoFile) :
    image, anno_names, anno_pts = _load_annotation(annoFile)
    for i in range(anno_names.shape[0]):
        anno_names[i] = anno_names[i].strip()
    imageFile = "{}/{}.jpg".format(VOC_IMAGE_ROOT, image)
    kps, desc = _compute_SIFT_features(imageFile, anno_pts)

    return anno_names, anno_pts, desc


def _load_keypoint_features(annoFile, feaType):

    # gen image patches for nodes
    imgfile, anno_names, anno_pts = _load_annotation(annoFile)
    imgPath = "{}/{}.jpg".format(VOC_IMAGE_ROOT, imgfile)
    descs = None
    patches = []
    feaType = "BBGM"
    if anno_names.shape[0]< 3:
        return anno_names, anno_pts, descs, patches, imgPath

    # strip space in names
    for i in range(anno_names.shape[0]):
        anno_names[i] = anno_names[i].strip()

    if feaType == VISFEA_TYPE_PRETRAINED_VGG16:
        descs = _load_cnn_feature(annoFile)
        # patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_BBGM:
        descs = _load_bbgm_feature(annoFile)
        # patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_PRETRAINED_LFNET:
        descs = _load_lfnet_feature(annoFile)
        patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
    elif feaType == VISFEA_TYPE_SIFT:
        kps, descs = GM_GenData._compute_SIFT_features(imgPath, anno_pts)
        # patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
        patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    elif feaType == VISFEA_TYPE_LFNET:
        kps, descs = GM_GenData._compute_SIFT_features(imgPath, anno_pts)
        patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
    elif feaType == VISFEA_TYPE_RAWPIXEL:
        patches = GM_GenData._compute_image_patches(imgPath, anno_pts, 16, 16)
        if anno_names.shape[0] > 0:
            descs = np.reshape(patches, (patches.shape[0], -1))
            descs = descs.astype(np.float64)


    return anno_names, anno_pts, descs, patches, imgPath


def _save_keypoint_features(annoFile):

    annoNames, annoPts, annoFeas = _compute_keypoint_features(annoFile)

    for i in range(annoNames.shape[0]):
        annoNames[i] = annoNames[i].strip()

    KeyPoints = {"annoNames": annoNames, "annoPts": annoPts, "annoFeas": annoFeas}

    path = os.path.splitext(annoFile)[0]
    splited = os.path.split(path)
    path = splited[0]
    file = splited[1]
    category = os.path.split(path)[1]

    fea_file = "{}/{}/{}.mat".format(VOC_FEAUTURES_ROOT, category, file)
    scipy.io.savemat(fea_file, KeyPoints)


    return 1

def _save_dataset_keyponit_features(dataset):
    for i in range(20):
        for xml_file in dataset.xml_list[i]:
            anno_file = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_file)
            _save_keypoint_features(anno_file)

    return 1

def _list_all_annotations():
    annoLists = []
    num_Files = 0
    for i in range(len(VOC_CATEGORIES)):
        annoFiles = _list_annotations(VOC_ANNOTATION_ROOT, VOC_CATEGORIES[i])

        annoLists.append(annoFiles)
        num_Files = num_Files + len(annoFiles)
    return annoLists

dataset_train = pascal_voc.PascalVOC('train', (256, 256))
dataset_test = pascal_voc.PascalVOC('test', (256, 256))

def _normalize_coordinates(points) :
    # normalize by center
    center = np.sum(points, axis = 0) / points.shape[0]
    norm_points = np.transpose(points)
    norm_points[0] = norm_points[0] - center[0]
    norm_points[1] = norm_points[1] - center[1]

    # normalized by max_distance
    distance = spatial.distance.cdist(points, points)
    maxDst = np.max(distance)
    norm_points = norm_points / maxDst

    points = np.transpose(norm_points)

    return points

def _swap(obj0, obj1):
    tmp = obj0
    obj0 = obj1
    obj1 = tmp
    return obj0, obj1

def _gen_features_VOC_augmentation(pts0, tails0, heads0,pts_fea0, patches0, pts1, tails1, heads1,pts_fea1, patches1, use_train_set):
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

    # if use_train_set:
    #     pts_fea0 = pts_fea0 * create_mask(pts_fea0.shape, probability=0.0)
    #     pts_fea1 = pts_fea1 * create_mask(pts_fea1.shape, probability=0.0)

    pts_fea0 = _mean_pooling(pts_fea0)
    pts_fea1 = _mean_pooling(pts_fea1)

    node_feaLen = pts_fea0.shape[1] + pts_fea1.shape[1]
    num_assGraph_nodes = num_matches
    senders, receivers, edge_features = [], [], []
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)

    for i in range(num_matches):
        cor_node0 = pts_fea0[gidx1[i]]
        cor_node1 = pts_fea1[gidx2[i]]
        node_features[i] = np.hstack((cor_node0, cor_node1))

    for i in range(num_edges0):
        cor_tail0 = pts0[tails0[i]]
        cor_head0 = pts0[heads0[i]]
        for k in range(num_edges1):

            # value = random.randint(0,100)
            # if value > 70 and use_train_set:
            #     continue

            cor_tail1 = pts1[tails1[k]]
            cor_head1 = pts1[heads1[k]]

            senders.append(tails0[i] * num_nodes1 + tails1[k])
            receivers.append(heads0[i] * num_nodes1 + heads1[k])
            edge_features.append(np.hstack((cor_tail0, cor_head0, cor_tail1, cor_head1)))

    senders = np.array(senders, np.int)
    receivers = np.array(receivers, np.int)
    edge_features = np.array(edge_features, np.float)

    assignGraph = {"gidx1": gidx1,
                   "gidx2": gidx2,
                   "node_features": node_features,
                   "senders": senders,
                   "receivers": receivers,
                   "edge_features": edge_features,
                   "patches1": patches0,
                   "patches2": patches1}
    return assignGraph

def _gen_features_VOC(pts0, tails0, heads0,pts_fea0, patches0, pts1, tails1, heads1,pts_fea1, patches1, use_train_set):
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

    if use_train_set:
        pts_fea0 = pts_fea0 * create_mask(pts_fea0.shape, probability=0.3)
        pts_fea1 = pts_fea1 * create_mask(pts_fea1.shape, probability=0.3)

    pts_fea0 = _mean_pooling(pts_fea0)
    pts_fea1 = _mean_pooling(pts_fea1)

    #node_feaLen = 512
    node_feaLen = pts_fea0.shape[1] + pts_fea1.shape[1]
    edge_feaLen = 8
    num_assGraph_nodes = num_matches
    num_assGraph_edges = num_edges0 * num_edges1
    senders = np.zeros(num_assGraph_edges, np.int)
    receivers = np.zeros(num_assGraph_edges, np.int)
    edge_features = np.zeros((num_assGraph_edges, edge_feaLen), np.float)
    node_features = np.zeros((num_assGraph_nodes, node_feaLen), np.float)

    for i in range(num_matches):
        cor_node0 = pts_fea0[gidx1[i]]
        cor_node1 = pts_fea1[gidx2[i]]
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
                    "edge_features": edge_features,
                    "patches1": patches0,
                    "patches2": patches1}
    return assignGraph

def _normalize_features(feature_maps):
    l2_norms = np.linalg.norm(feature_maps, axis=-1, keepdims=True)

    return feature_maps / l2_norms

# def _topk_index(mat, topk):
#     num_rows = mat.shape[0]
#     cols = np.argsort(mat, axis=1)[:, -topk:].flatten()
#     # cols = np.array([list(range(topk)) for _ in range(num_rows)], np.int).flatten()
#     rows = np.array(range(num_rows)).repeat(topk)
#
#     return rows, cols

def _row_col_dict(rows, cols):
    num_idx = len(rows)
    rc_dict = {}
    for idx in range(num_idx):
        if not rows[idx] in rc_dict.keys():
            rc_dict[rows[idx]] = [cols[idx]]
        else:
            if not cols[idx] in rc_dict[rows[idx]]:
                rc_dict[rows[idx]].append(cols[idx])

    return rc_dict

def _topk_index(mat, topk):
    num_rows, num_cols = mat.shape[0], mat.shape[1]
    cols12 = np.argsort(mat, axis=1)[:, -topk:].flatten()
    rows12 = np.array(range(num_rows)).repeat(topk)

    rows21 = np.argsort(mat, axis=0)[:, -topk:].flatten()
    cols21 = np.array(range(num_cols)).repeat(topk)

    rows = rows12.tolist() + rows21.tolist()
    cols = cols12.tolist() + cols21.tolist()

    rc_dict = _row_col_dict(rows, cols)

    rows_list, cols_list = [], []
    for row, col_list in rc_dict.items():
        for col in col_list:
            rows_list.append(row)
            cols_list.append(col)

    return np.array(rows_list), np.array(cols_list)

def _gen_edge_dict(tails, heads):
    num_edges = len(tails)
    edge_dict = dict()
    for idx in range(num_edges):
        tail = tails[idx]
        head = heads[idx]
        if not tail in edge_dict.keys():
            edge_dict[tail] = [head]
        else:
            edge_dict[tail].append(head)

    return edge_dict

def _gen_features_VOC_topk(pts0, tails0, heads0,pts_fea0, patches0, pts1, tails1, heads1,pts_fea1, patches1, topk = 5):

    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    tails0_sup = heads0_sup = np.array(range(num_nodes0), np.int)
    tails1_sup = heads1_sup = np.array(range(num_nodes1), np.int)
    tails0 = np.concatenate((tails0, tails0_sup), axis=-1)
    tails1 = np.concatenate((tails1, tails1_sup), axis=-1)
    heads0 = np.concatenate((heads0, heads0_sup), axis=-1)
    heads1 = np.concatenate((heads1, heads1_sup), axis=-1)

    topk = min(topk, num_nodes1)

    edge_dict0, edge_dict1 = _gen_edge_dict(tails0, heads0), _gen_edge_dict(tails1, heads1)

    similarity = np.matmul(_normalize_features(pts_fea0), _normalize_features(pts_fea1).transpose((1, 0)))
    rows, cols = _topk_index(similarity, topk)
    gidx1, gidx2 = rows, cols
    num_matches = len(rows)

    pts_fea0 = pts_fea0 * create_mask(pts_fea0.shape, probability=0.3)
    pts_fea0 = _mean_pooling(pts_fea0)
    pts_fea1 = pts_fea1 * create_mask(pts_fea1.shape, probability=0.3)
    pts_fea1 = _mean_pooling(pts_fea1)

    node_feaLen = pts_fea0.shape[1] + pts_fea1.shape[1]
    senders, receivers, edge_features = [], [], []  # np.int
    node_features = np.zeros((num_matches, node_feaLen), np.float)

    for i in range(num_matches):
        cor_node0 = pts_fea0[gidx1[i]]
        cor_node1 = pts_fea1[gidx2[i]]
        node_features[i] = np.hstack((cor_node0, cor_node1))

    for ii in range(num_matches):
        tail0, tail1 = gidx1[ii], gidx2[ii]
        for jj in range(num_matches):
            # if ii == jj:
            #     continue

            head0, head1 = gidx1[jj], gidx2[jj]
            if head0 in edge_dict0[tail0] and head1 in edge_dict1[tail1]:
                senders.append(ii)
                receivers.append(jj)

                cor_tail0 = pts0[tail0]
                cor_head0 = pts0[head0]
                cor_tail1 = pts1[tail1]
                cor_head1 = pts1[head1]

                edge_features.append(np.hstack((cor_tail0, cor_head0, cor_tail1, cor_head1)))

    senders = np.array(senders, np.int)
    receivers = np.array(receivers, np.int)
    num_edges = len(edge_features)
    edge_features = np.array(edge_features, np.float)

    assignGraph = {"gidx1": gidx1,
                    "gidx2": gidx2,
                    "node_features": node_features,
                    "senders": senders,
                    "receivers": receivers,
                    "edge_features": edge_features,
                    "patches1": patches0,
                    "patches2": patches1}
    return assignGraph, num_edges > 0

def _remove_unmatched_points(anno_names0, anno_pts0, anno_descs0, patches0,
                             anno_names1, anno_pts1, anno_descs1, patches1):
    valid0 = np.zeros(anno_names0.shape[0], dtype = np.bool)
    valid1 = np.zeros(anno_names1.shape[0], dtype = np.bool)

    for i in range(anno_names0.shape[0]):
        for k in range(anno_names1.shape[0]):
            if anno_names0[i] == anno_names1[k]:
                valid0[i] = True
                valid1[k] = True
                break

    anno_names0 = anno_names0[valid0]
    anno_pts0 = anno_pts0[valid0]
    anno_descs0 = anno_descs0[valid0]
    patches0 = patches0[valid0]

    anno_names1 = anno_names1[valid1]
    anno_pts1 = anno_pts1[valid1]
    anno_descs1 = anno_descs1[valid1]
    patches1 = patches1[valid1]

    return anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1


def _gen_affinity_VOC(lib,
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

    count = lib.build_affinity_VOC(
        num_nodes0, num_edges0, tails0, heads0, dists0, angs0,
        num_nodes1, num_edges1, tails1, heads1, dists1, angs1,
        gidx1, gidx2, K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    return gidx1, gidx2, K

def pooling_mean(pts_fea):
    n_pts, dim_feat = pts_fea.shape
    if n_pts == 0:
        return None
    pts_fea = pts_fea.reshape(-1, int(MEAN_POOLING_INTERVAL_PASCAL))
    pts_fea = np.mean(pts_fea, axis=1)
    pts_fea = pts_fea.reshape((n_pts, -1))
    return pts_fea

def _gen_random_graph(rand,
                      lib,
                      use_train_set,
                      category_id,
                      num_outlier_min_max,
                      feaType):

#    t0 = time.time()
    while True:
        if use_train_set:
            xml_files = dataset_train.get_xml_pair(category_id)
        else:
            xml_files = dataset_test.get_xml_pair(category_id)
        xml_file0 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[0])
        xml_file1 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[1])

        anno_names0, anno_pts0, anno_descs0, patches0, imgPath0 = _load_keypoint_features(xml_file0, feaType)
        anno_names1, anno_pts1, anno_descs1, patches1, imgPath1 = _load_keypoint_features(xml_file1, feaType)

        if anno_pts0.shape[0] < 10 or anno_pts1.shape[0] < 10:
            continue

        # remove unmatched points
        anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1 = _remove_unmatched_points(
            anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1)


        if anno_pts0.shape[0] >= 10 and anno_pts1.shape[0] >= 10:
            break

    pts0 = anno_pts0.copy()
    pts1 = anno_pts1.copy()
    descs0 = anno_descs0.copy()
    descs1 = anno_descs1.copy()
    names0 = anno_names0.copy()
    names1 = anno_names1.copy()

    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    # randomly re-order
    # index0 = np.arange(0, pts0.shape[0])
    # rand.shuffle(index0)
    # pts0 = pts0[index0]
    # descs0 = descs0[index0]
    # names0 = names0[index0]
    # patches0 = patches0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    descs1 = descs1[index1]
    names1 = names1[index1]
    patches1 = patches1[index1]

    matchInfo = {"pts1": pts0.copy(),
                     "pts2": pts1.copy()}

    if GM_GenData.GRAPH_MODE_PASCAL == "DEL":
        tails0, heads0, dists0, angs0 = GM_GenData._build_delaunay_graph(pts0)
        tails1, heads1, dists1, angs1 = GM_GenData._build_delaunay_graph(pts1)
    elif GM_GenData.GRAPH_MODE_PASCAL == "KNN":
        tails0, heads0, dists0, angs0 = GM_GenData._build_knn_graph(pts0, GM_GenData.NUM_K_PASCAL)
        tails1, heads1, dists1, angs1 = GM_GenData._build_knn_graph(pts1, GM_GenData.NUM_K_PASCAL)

    # descs0 = backbone(descs0, tails0, heads0, pts0)
    # descs1 = backbone(descs1, tails1, heads1, pts1)

    descs0 = pooling_mean(descs0)
    descs1 = pooling_mean(descs1)

    # normalize point coordinates
    pts0 = _normalize_coordinates(pts0)
    pts1 = _normalize_coordinates(pts1)


    # if num_nodes1 < num_nodes0:
    #     num_sup = num_nodes0 - num_nodes1
    #     descs1_sup = np.zeros((num_sup, descs1.shape[1]))
    #     pts_sup = np.zeros((num_sup, 2))
    #     names_sup = np
    #     descs1 = np.concatenate([descs1, descs1_sup], axis=0)
    #     pts1 = np.concatenate([pts1, pts_sup], axis=0)

    # record ground-truth matches
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(anno_pts0.shape[0]):
        for k in range(anno_pts1.shape[0]):
            if names0[i] == names1[k]:
                gX[i][k] = 1.0
                break

    assignGraph = _gen_features_VOC_augmentation(pts0, tails0, heads0,descs0, patches0, pts1, tails1, heads1,descs1, patches1, use_train_set)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": VOC_CATEGORIES[category_id],
             "image1": imgPath0,
             "image2": imgPath1}

    matchInfo["tails1"] = tails0.copy()
    matchInfo["heads1"] = heads0.copy()
    matchInfo["tails2"] = tails1.copy()
    matchInfo["heads2"] = heads1.copy()
    matchInfo["gidx1"] = gidx1.copy()
    matchInfo["gidx2"] = gidx2.copy()

    affinity=None
    # affinity = {"gidx1": aff_gidx1,
    #             "gidx2": aff_gidx2,
    #             "K": aff_K}

    return assignGraph, image, matchInfo, affinity
    

def _random_index(rand, sum_rates):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    randnum = rand.randint(1, sum_rates[-1])
    for index in range(len(sum_rates)):
        if randnum <= sum_rates[index]:
            break
    return index


def gen_random_graphs_VOC(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max,
                              feaType,
                              use_train_set = True,
                              category_id = -1):
    lib=None

    graphs = []
    images = []
    matchInfos = []
    affinities = []
    for _ in range(num_examples):
        if category_id < 0:
            #cid = rand.randint(0, 20)
            # cid = rand.randint(0, len(VOC_CATEGORIES))
            cid = 1
        else:
            cid = category_id
#        cid = RATE_INDEXS[rand.randint(0, SUM_RATES)]
        graph, image, matchInfo, affinity = _gen_random_graph(rand,
                                         lib,
                                         use_train_set,
                                         cid,
                                         num_outlier_min_max=num_outlier_min_max,
                                         feaType = feaType)
        graphs.append(graph)
        images.append(image)
        matchInfos.append(matchInfo)
        affinities.append(affinity)

    return graphs, images, matchInfos, affinities

# seed = 66
# rand = np.random.RandomState(seed=seed)
# lib = None
#
# graph, image, matchInfo, affinity = _gen_random_graph(rand,
#                                          lib,
#                                          True,
#                                          rand.randint(0, len(VOC_CATEGORIES)),
#                                          num_outlier_min_max=(10,11),
#                                          feaType = "BBGM")