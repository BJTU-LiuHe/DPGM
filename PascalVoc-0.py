import numpy as np
import torch
from scipy import spatial
from scipy.spatial import Delaunay

from ctypes import *
import numpy.ctypeslib as npct

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
# from torch_geometric.data import Data
# from NGMv2 import Net
#
# backbone = Net()
# pretrained_backbone = torch.load("pretrained/pretrained_params_vgg16_ngmv2_voc.pt")
# state_dict = backbone.state_dict()
# backbone.load_state_dict(pretrained_backbone, strict=False)
#
# if torch.cuda.is_available():
#     backbone.cuda()
# backbone.eval()

ROOT = "/home/6T/lh/data/graph_matching/"

VOC_CNN_FEA_ROOT=ROOT + 'cnn_features_mat/PascalVoc'
VOC_LFNET_FEA_ROOT=ROOT + 'lfnet_features/PascalVoc'
VOC_ANNOTATION_ROOT = ROOT + "PascalVoc/annotations"
VOC_FEAUTURES_ROOT = ROOT + "PascalVoc/annotations_features"
VOC_BBGM_FEATURE_ROOT = ROOT + "BBGM_features/PascalVoc"
VOC_IMAGE_ROOT = ROOT + "PascalVoc/JPEGImages"


VOC_CATEGORIES = ["aeroplane",   "bicycle", "bird",  "boat",      "bottle", "bus",         "car",   "cat",   "chair", "cow",
                  "diningtable", "dog",     "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",  "train", "tvmonitor"]
#VOC_CATEGORY_INNERS = [16,        11,         12,        11,          8,       8,             13,     16,      10,      16,
#                       8,         16,         16,        10,         20,       6,             16,     12,      7,       8]
#VOC_CATEGORIES = ["bottle", "diningtable" , "pottedplant", "sofa", "train", "tvmonitor"]
#VOC_CATEGORIES = ["bottle",  "diningtable" , "pottedplant", "train"]
#VOC_CATEGORIES = ["bottle"]
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
    # path = os.path.splitext(annoFile)[0]
    # splited = os.path.split(path)
    # path = splited[0]
    # file = splited[1]
    # category = os.path.split(path)[1]
    #
    # fea_file = "{}/{}/{}.mat".format(VOC_FEAUTURES_ROOT, category, file)
    # KeyPoints = scipy.io.loadmat(fea_file)
    #
    # annoNames = KeyPoints["annoNames"]
    # for i in range(annoNames.shape[0]):
    #     annoNames[i] = annoNames[i].strip()
    #
    # KeyPoints["annoNames"] = annoNames

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


    # imgFile = "{}/{}.jpg".format(VOC_IMAGE_ROOT, file)
    # image = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    # patches = np.zeros(shape = (pts.shape[0], KEYPOINT_PATCH_WIDTH, KEYPOINT_PATCH_HEIGHT), dtype = image.dtype)
    # for i in range(pts.shape[0]):
    #     patch = GM_GenData._extract_img_patch(image, pts[i][0], pts[i][1], KEYPOINT_PATCH_WIDTH, KEYPOINT_PATCH_HEIGHT)
    #     patches[i] = patch
    #
    # patches = patches.astype(np.float32) / 255.0

    # return  KeyPoints["annoNames"], KeyPoints["annoPts"], KeyPoints["annoFeas"], patches

    return anno_names, anno_pts, descs, patches, imgPath

# def _load_keypoint_features(annoFile):
#     path = os.path.splitext(annoFile)[0]
#     splited = os.path.split(path)
#     path = splited[0]
#     file = splited[1]
#
#     category = os.path.split(path)[1]
#
#     fea_file = "{}/{}/{}.mat".format(VOC_FEAUTURES_ROOT, category, file)
#     KeyPoints = scipy.io.loadmat(fea_file)
#
#     annoNames = KeyPoints["annoNames"]
#     for i in range(annoNames.shape[0]):
#         annoNames[i] = annoNames[i].strip()
#     KeyPoints["annoNames"] = annoNames
#
#     # load cnn features
#     cnn_file=VOC_CNN_FEA_ROOT+'/'+path.split('/')[-1]+'/'+file+'.pth'
#     cnn_infos=t.load(cnn_file)  #pts,pts_features
#     if len(cnn_infos['pts'])==0:
#         return KeyPoints["annoNames"], KeyPoints["annoPts"], np.zeros((0,0))
#     else:
#         return  KeyPoints["annoNames"], KeyPoints["annoPts"], cnn_infos['pts_features']

def _save_keypoint_features(annoFile):
#    gray, annoNames, annoPts = _read_image_features(annoFile)
#
#    annoFeas = np.zeros((annoPts.shape[0], 128), np.float)
#    for i in range(annoPts.shape[0]):
#        mag, ori, annoFeas[i] = Siftor.calcPointSift(gray, np.int(annoPts[i][0]), np.int(annoPts[i][1]))

    annoNames, annoPts, annoFeas = _compute_keypoint_features(annoFile)

    for i in range(annoNames.shape[0]):
        annoNames[i] = annoNames[i].strip()

    # for i in range(annoNames.shape[0]):
    #     msg = 'len({}) = {}'.format(annoNames[i], len(annoNames[i]))
    #     print(msg)

    KeyPoints = {"annoNames": annoNames, "annoPts": annoPts, "annoFeas": annoFeas}

    path = os.path.splitext(annoFile)[0]
    splited = os.path.split(path)
    path = splited[0]
    file = splited[1]
    category = os.path.split(path)[1]

    fea_file = "{}/{}/{}.mat".format(VOC_FEAUTURES_ROOT, category, file)
    # print(annoFile)
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

        # for k in range(len(annoFiles)):
        #     _save_keypoint_features(annoFiles[k])

        # for k in range(len(annoFiles)):
        #     annoNames, annoPts = _read_image_features(annoFiles[k])
        #     if len(annoNames) < VOC_CATEGORY_INNERS[i] :
        #         os.remove(annoFiles[k])

        annoLists.append(annoFiles)
        num_Files = num_Files + len(annoFiles)
    return annoLists


#SUM_RATES, RATE_INDEXS = _compute_rate_indexs()

dataset_train = pascal_voc.PascalVOC('train', (256, 256))
dataset_test = pascal_voc.PascalVOC('test', (256, 256))
#_save_dataset_keyponit_features(dataset_train)
#_save_dataset_keyponit_features(dataset_test)

#VOC_ANNO_LISTS = _list_all_annotations()

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

    # # normalize by deviation
    # deviation = np.nanstd(norm_points, axis=1)
    # norm_points[0] = norm_points[0] / deviation[0]
    # norm_points[1] = norm_points[1] / deviation[1]

    points = np.transpose(norm_points)

    return points

# def _build_threshold_graph(points, threshold) :
#     distance = spatial.distance.cdist(points, points)
#
#     idxs = np.where(distance <= threshold)
#
#     dists = distance[idxs]
#     tails = idxs[0]
#     heads = idxs[1]
#
#     edges = points[heads] - points[tails]
#     angs = np.zeros(edges.shape[0], dtype = np.float)
#     for i in range(edges.shape[0]):
#     #    angs[i] = math.atan2(edges[i][1], edges[i][0])
#         angs[i] = math.fabs(math.atan(edges[i][1] / (edges[i][0] + 1e-16)))
#
#     return tails, heads, dists, angs
#
# def _build_delaunay_graph(points) :
#     A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)
#     distance = spatial.distance.cdist(points, points)
#
#     if points.shape[0] < 3:
#         A = distance
#     else:
#         triangles = Delaunay(points).simplices
#         for tri in triangles:
#             A[tri[0]][tri[1]] = distance[tri[0]][tri[1]]
#             A[tri[0]][tri[2]] = distance[tri[0]][tri[2]]
#             A[tri[1]][tri[2]] = distance[tri[1]][tri[2]]
#             A[tri[1]][tri[0]] = A[tri[0]][tri[1]]
#             A[tri[2]][tri[0]] = A[tri[0]][tri[2]]
#             A[tri[2]][tri[1]] = A[tri[1]][tri[2]]
#
#     idxs = np.nonzero(A)
#     dists = A[idxs]
#     tails = idxs[0]
#     heads = idxs[1]
#
#     edges = points[heads] - points[tails]
#     angs = np.zeros(edges.shape[0], dtype = np.float)
#     for i in range(edges.shape[0]):
#     #    angs[i] = math.atan2(edges[i][1], edges[i][0])
#         angs[i] = math.fabs(math.atan(edges[i][1] / (edges[i][0] + 1e-16)))
#
#     return tails, heads, dists, angs

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

# def _compute_feature_similarity(descs0, descs1):
#     # normalize features
#     norm0 = np.linalg.norm(descs0, axis = 1) + 1e-16
#     norm1 = np.linalg.norm(descs1, axis = 1) + 1e-16
#     for i in range(descs0.shape[0]) :
#         descs0[i][:] = descs0[i][:] / norm0[i]
#     for i in range(descs1.shape[0]) :
#         descs1[i][:] = descs1[i][:] / norm1[i]
#
#     # compute feature similarity via vector cosine
#     fea_sim = np.matmul(descs0, np.transpose(descs1))
#
#     return fea_sim

# def _compute_node_affinity(imgGray0, pts0, imgGray1, pts1):
#     n0 = pts0.shape[0]
#     n1 = pts1.shape[0]
#     ori0 = np.zeros(n0, np.float)
#     ori1 = np.zeros(n1, np.float)
#     desc0 = np.zeros((n0, 128), np.int)
#     desc1 = np.zeros((n1, 128), np.int)
#
#     for i in range(n0):
#         _, ori0[i], desc0[i] = Siftor.calcPointSift(imgGray0, np.int(pts0[i][0]), np.int(pts0[i][1]))
#     for i in range(n1):
#         _, ori1[i], desc1[i] = Siftor.calcPointSift(imgGray1, np.int(pts1[i][0]), np.int(pts1[i][1]))
#
#     desc0 = desc0.astype(np.float)
#     desc1 = desc1.astype(np.float)
#     affinity = _compute_feature_similarity(desc0, desc1)
#
#
#     # for i in range(n0):
#     #     for k in range(n1):
#     #         ang = math.fabs(ori0[i] - ori1[k])
#     #         if ang > math.pi:
#     #             ang = math.fabs(2 * math.pi - ang)
#     #         affinity[i][k] = math.exp(-ang)
#
#     return affinity

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

def _swap(obj0, obj1):
    tmp = obj0
    obj0 = obj1
    obj1 = tmp
    return obj0, obj1

def _gen_features_VOC(pts0, tails0, heads0,pts_fea0, patches0, pts1, tails1, heads1,pts_fea1, patches1):
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
        # cor_node0 = pts0[gidx1[i]]
        # cor_node1 = pts1[gidx2[i]]
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

# def _gen_random_affinity(rand,
#                       lib,
#                       use_train_set,
#                       category_id,
#                       num_outlier_min_max):
#     while True:
#         if use_train_set:
#             xml_files = dataset_train.get_xml_pair(category_id)
#         else:
#             xml_files = dataset_test.get_xml_pair(category_id)
#         xml_file0 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[0])
#         xml_file1 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[1])
#
#     #    anno_names0, anno_pts0, anno_descs0 = _load_keypoint_features(xml_file0)
#     #    anno_names1, anno_pts1, anno_descs1 = _load_keypoint_features(xml_file1)
#
#         anno_names0, anno_pts0, anno_descs0 = _compute_keypoint_features(xml_file0)
#         anno_names1, anno_pts1, anno_descs1 = _compute_keypoint_features(xml_file1)
#
#
#
#         # remove unmatched points
#         anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1 = _remove_unmatched_points(
#             anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1)
#
#         if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
#             break
#
#     pts0 = anno_pts0.copy()
#     pts1 = anno_pts1.copy()
#     descs0 = anno_descs0.copy()
#     descs1 = anno_descs1.copy()
#     names0 = anno_names0.copy()
#     names1 = anno_names1.copy()
#
#     # randomly re-order
#     # index0 = np.arange(0, pts0.shape[0])
#     # rand.shuffle(index0)
#     # pts0 = pts0[index0]
#     # descs0 = descs0[index0]
#     # names0 = names0[index0]
#
#     index1 = np.arange(0, pts1.shape[0])
#     rand.shuffle(index1)
#     pts1 = pts1[index1]
#     descs1 = descs1[index1]
#     names1 = names1[index1]
#
#     # normalize point coordinates
#     pts0 = _normalize_coordinates(pts0)
#     pts1 = _normalize_coordinates(pts1)
#
#     # record ground-truth matches
#     gX = np.zeros((pts0.shape[0], pts1.shape[0]))
#     for i in range(anno_pts0.shape[0]):
#         for k in range(anno_pts1.shape[0]):
#             if names0[i] == names1[k]:
#                 gX[i][k] = 1.0
#                 break
#
#     tails0, heads0, dists0, angs0 = _build_delaunay_graph(pts0)
#     tails1, heads1, dists1, angs1 = _build_delaunay_graph(pts1)
#     num_nodes0 = pts0.shape[0]
#     num_nodes1 = pts1.shape[0]
#     gidx1, gidx2, K = _gen_affinity_VOC(lib, num_nodes0, tails0, heads0, dists0, angs0,
#                                         num_nodes1, tails1, heads1, dists1, angs1)
#
#     solutions = np.zeros(len(gidx1), np.bool)
#     for i in range(len(gidx1)) :
#         if gX[gidx1[i]][gidx2[i]] :
#             solutions[i] = True
#
#
#     affinity = {"gidx1": gidx1,
#                 "gidx2": gidx2,
#                 "K": K,
#                 "solutions": solutions}
#     image = {"category": VOC_CATEGORIES[category_id],
#              "image1": xml_file0,
#              "image2": xml_file1}
#
#     return affinity, image


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
        #anno_names0, anno_pts0, anno_descs0 = _compute_keypoint_features(xml_file0)
        #anno_names1, anno_pts1, anno_descs1 = _compute_keypoint_features(xml_file1)

        if anno_pts0.shape[0] < 3 or anno_pts1.shape[0] < 3:
            continue

        # remove unmatched points
        anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1 = _remove_unmatched_points(
            anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1)


        if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
            break


    pts0 = anno_pts0.copy()
    pts1 = anno_pts1.copy()
    descs0 = anno_descs0.copy()
    descs1 = anno_descs1.copy()
    names0 = anno_names0.copy()
    names1 = anno_names1.copy()

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

    # record ground-truth matches
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(anno_pts0.shape[0]):
        for k in range(anno_pts1.shape[0]):
            if names0[i] == names1[k]:
                gX[i][k] = 1.0
                break


    assignGraph = _gen_features_VOC(pts0, tails0, heads0,descs0, patches0, pts1, tails1, heads1,descs1, patches1)


    num_nodes0 = pts0.shape[0]
    num_nodes1 = pts1.shape[0]
    # aff_gidx1, aff_gidx2, aff_K = _gen_affinity_VOC(lib, num_nodes0, tails0, heads0, dists0, angs0,
    #                                     num_nodes1, tails1, heads1, dists1, angs1)


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
    
"""
def _gen_random_graph(rand,
                      use_train_set,
                      category_id,
                      num_outlier_min_max,
                      record_file=None,
                      feaType ="BBGM"):
    category = VOC_CATEGORIES[category_id]

    if record_file is not None:
        xml_name0, xml_name1 = record_file[:-4].split("-")
        xml_file0 = "{}/{}/{}.xml".format(VOC_ANNOTATION_ROOT, category, xml_name0)
        xml_file1 = "{}/{}/{}.xml".format(VOC_ANNOTATION_ROOT, category,xml_name1)
        #   anno_names0, anno_pts0, anno_descs0 = _load_keypoint_features(xml_file0)
        #   anno_names1, anno_pts1, anno_descs1 = _load_keypoint_features(xml_file1)
        anno_names0, anno_pts0, anno_descs0, patches0, imgPath0 = _load_keypoint_features(xml_file0, feaType)
        anno_names1, anno_pts1, anno_descs1, patches1, imgPath1 = _load_keypoint_features(xml_file1, feaType)

        # remove unmatched points
        #    anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1 = _remove_unmatched_points(
        #        anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1)
        anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1 = _remove_unmatched_points(
            anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1)

    else:
        while True:
            if use_train_set:
                xml_files = dataset_train.get_xml_pair(category_id)
            else:
                xml_files = dataset_test.get_xml_pair(category_id)
            xml_file0 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[0])
            xml_file1 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[1])
            anno_names0, anno_pts0, anno_descs0, patches0, imgPath0 = _load_keypoint_features(xml_file0, feaType)
            anno_names1, anno_pts1, anno_descs1, patches1, imgPath1 = _load_keypoint_features(xml_file1, feaType)
            # image0, anno_names0, anno_pts0 = _load_annotation(xml_file0)
            # image1, anno_names1, anno_pts1 = _load_annotation(xml_file1)

            # remove unmatched points
            #    anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1 = _remove_unmatched_points(
            #        anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1)
            anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1 = _remove_unmatched_points(
                anno_names0, anno_pts0, anno_descs0, patches0, anno_names1, anno_pts1, anno_descs1, patches1)

            if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
                break

    pts0 = anno_pts0.copy()
    pts1 = anno_pts1.copy()
    descs0 = anno_descs0.copy()
    descs1 = anno_descs1.copy()
    names0 = anno_names0.copy()
    names1 = anno_names1.copy()

    # randomly re-order
    # index0 = np.arange(0, pts0.shape[0])
    # rand.shuffle(index0)
    # pts0 = pts0[index0]
    # descs0 = descs0[index0]
    # names0 = names0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    descs1 = descs1[index1]
    names1 = names1[index1]

    org_pts0, org_pts1 = pts0, pts1
    # normalize point coordinates
    pts0 = _normalize_coordinates(pts0)
    pts1 = _normalize_coordinates(pts1)

    # record ground-truth matches
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(anno_pts0.shape[0]):
        for k in range(anno_pts1.shape[0]):
            if names0[i] == names1[k]:
                gX[i][k] = 1.0
                break

    tails0, heads0, dists0, angs0 = GM_GenData._build_delaunay_graph(pts0)
    tails1, heads1, dists1, angs1 = GM_GenData._build_delaunay_graph(pts1)
    #
    # tails0, heads0 = semantic_adj(names0, VOC_CATEGORIES[category_id])
    # tails1, heads1 = semantic_adj(names1, VOC_CATEGORIES[category_id])

    assignGraph = _gen_features_VOC(pts0, tails0, heads0,descs0, patches0, pts1, tails1, heads1,descs1, patches1)

    adj0, adj1 = np.zeros((pts0.shape[0],pts0.shape[0]),dtype=np.int), np.zeros((pts1.shape[0],pts1.shape[0]),dtype=np.int)
    for idx0 in range(len(tails0)):
        adj0[tails0[idx0], heads0[idx0]] = 1

    for idx1 in range(len(tails1)):
        adj1[tails1[idx1], heads1[idx1]] = 1

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": VOC_CATEGORIES[category_id],
             "image1": xml_file0,
             "image2": xml_file1}

    attr_dict ={"pts0": org_pts0,
                "pts1": org_pts1,
                "graph0": adj0,
                "graph1": adj1,
                "X_gt": gX}
    return assignGraph, image, attr_dict
"""

def _random_index(rand, sum_rates):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    randnum = rand.randint(1, sum_rates[-1])
    for index in range(len(sum_rates)):
        if randnum <= sum_rates[index]:
            break
    return index

# def gen_random_affinities_VOC(rand,
#                               num_examples,
#                               num_inner_min_max,
#                               num_outlier_min_max,
#                               category_id = -1):
#
#     lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
#     lib.build_affinity_VOC.argtypes = [c_int,
#                                             c_int,
#                                             npct.ndpointer(dtype = np.int, ndim = 1, flags = "C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                             c_int,
#                                             c_int,
#                                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                             npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]
#     lib.build_affinity_VOC2.argtypes = [c_int,
#                                              c_int,
#                                              npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                              npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
#                                              c_int,
#                                              npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                              npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
#                                              c_float,
#                                              npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]
#
#     affinities = []
#     images = []
#     for _ in range(num_examples):
#         if category_id < 0:
#             #cid = rand.randint(0, 20)
#             cid = rand.randint(0, len(VOC_CATEGORIES))
#         else:
#             cid = category_id
# #        cid = RATE_INDEXS[rand.randint(0, SUM_RATES)]
#         affinity, image = _gen_random_affinity(rand,
#                                         lib,
#                                         use_train_set = 0,
#                                         category_id = cid,
#                                         num_outlier_min_max=num_outlier_min_max)
#         affinities.append(affinity)
#         images.append(image)
#
#
#     return affinities, images

def gen_random_graphs_VOC(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max,
                              feaType,
                              use_train_set = True,
                              category_id = -1):
    lib=None
    # lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
    # lib.build_affinity_VOC.argtypes = [c_int,
    #                                    c_int,
    #                                    npct.ndpointer(dtype = np.int, ndim = 1, flags = "C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                    c_int,
    #                                    c_int,
    #                                    npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                    npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]

    graphs = []
    images = []
    matchInfos = []
    affinities = []
    for _ in range(num_examples):
        if category_id < 0:
            #cid = rand.randint(0, 20)
            cid = rand.randint(0, len(VOC_CATEGORIES))
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

"""

def gen_random_graphs_VOC(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max,
                              use_train_set = True,
                              category_id = -1,
                              record_file = None,
                              feaType="BBGM"):

    graphs = []
    images = []
    attri_dicts = []
    for _ in range(num_examples):
        if category_id < 0:
            #cid = rand.randint(0, 20)
            cid = rand.randint(0, len(VOC_CATEGORIES))
        else:
            cid = category_id
#        cid = RATE_INDEXS[rand.randint(0, SUM_RATES)]
        graph, image, attri_dict = _gen_random_graph(rand,
                                         use_train_set,
                                         cid,
                                         num_outlier_min_max=num_outlier_min_max,
                                         record_file = record_file)
        graphs.append(graph)
        images.append(image)
        attri_dicts.append(attri_dict)
    return graphs, images, attri_dicts
"""