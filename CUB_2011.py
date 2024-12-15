import scipy.io
import random

import GM_GenData
from GM_GenData import *

ROOT = "/home/lh/data/CUB_2011_features/"
TRAIN_DATA_ROOT = os.path.join(ROOT, "train")
TEST_DATA_ROOT = os.path.join(ROOT, "test")
CATEGORIES = os.listdir(TRAIN_DATA_ROOT)
CLS_IMAGES_TRAIN = GM_GenData._map_cls_to_images(TRAIN_DATA_ROOT, CATEGORIES)
CLS_IMAGES_TEST = GM_GenData._map_cls_to_images(TEST_DATA_ROOT, CATEGORIES)

def _load_mat_file(matfile):
    matData = scipy.io.loadmat(matfile)
    anno_pts = matData["kpts"]
    descs = matData["features"].transpose()
    patches = np.zeros(shape=(anno_pts.shape[0], 1, 1), dtype=np.uint8)
    names = matData["labels"].reshape((anno_pts.shape[0],))

    return anno_pts, descs, patches, names

def _normalize_features(feature_maps):
    l2_norms = np.linalg.norm(feature_maps, axis=-1, keepdims=True)

    return feature_maps / l2_norms


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

def _mean_pooling(pts_fea):

    n_pts, dim_feat = pts_fea.shape
    pts_fea = pts_fea.reshape(-1, int(MEAN_POOLING_INTERVAL_CUB))
    pts_fea = np.mean(pts_fea, axis=1)
    pts_fea = pts_fea.reshape((n_pts, -1))

    return pts_fea

def create_mask(shape, probability):
    prob_value = probability * 1000
    mask = np.random.randint(1, 1000, size= shape)*1.0
    mask[mask < prob_value] = 0
    mask[mask >= prob_value] = 1

    return mask

def _gen_features_CUB(pts0, tails0, heads0,pts_feas0, patches0, pts1, tails1, heads1,pts_feas1, patches1, use_train_set):
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

    if use_train_set:
        pts_feas0 = pts_feas0 * create_mask(pts_feas0.shape, probability=0.3)
        pts_feas1 = pts_feas1 * create_mask(pts_feas1.shape, probability=0.3)

    pts_feas0 = _mean_pooling(pts_feas0)
    pts_feas1 = _mean_pooling(pts_feas1)

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

def _gen_random_graph(rand,
                      lib,
                      category,
                      num_outlier_min_max,
                      feaType,
                      use_train_set = True) :
    while True:
        if use_train_set:
            imgFiles = CLS_IMAGES_TRAIN[category]
            matFiles = random.sample(imgFiles, 2)
            mat_path = os.path.join(TRAIN_DATA_ROOT, category)
        else:
            imgFiles = CLS_IMAGES_TEST[category]
            matFiles = random.sample(imgFiles, 2)
            mat_path = os.path.join(TEST_DATA_ROOT, category)
        anno_pts0, anno_descs0, patches0, names0 = _load_mat_file(os.path.join(mat_path, matFiles[0]))
        anno_pts1, anno_descs1, patches1, names1 = _load_mat_file(os.path.join(mat_path, matFiles[1]))

        if anno_pts0.shape[0] < 3 or anno_pts1.shape[0] < 3:
            continue

        # remove unmatched points
        names0, anno_pts0, anno_descs0, patches0, names1, anno_pts1, anno_descs1, patches1 = _remove_unmatched_points(
            names0, anno_pts0, anno_descs0, patches0, names1, anno_pts1, anno_descs1, patches1)

        if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
            break

    pts0 = anno_pts0
    pts1 = anno_pts1
    descs0 = anno_descs0
    descs1 = anno_descs1

    # randomly re-order
    index0 = np.arange(0, pts0.shape[0])
    rand.shuffle(index0)
    pts0 = pts0[index0]
    descs0 = descs0[index0]
    names0 = names0[index0]
    if patches0 is not None:
        patches0 = patches0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    descs1 = descs1[index1]
    names1 = names1[index1]
    if patches1 is not None:
        patches1 = patches1[index1]

    matchInfo = {"pts1": pts0.copy(),
                 "pts2": pts1.copy()}


    # normalize point coordinates
    pts0 = GM_GenData._normalize_coordinates(pts0)
    pts1 = GM_GenData._normalize_coordinates(pts1)

    # record ground-truth matches
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(anno_pts0.shape[0]):
        for k in range(anno_pts1.shape[0]):
            if names0[i] == names1[k]:
                gX[i][k] = 1.0
                break

    if GM_GenData.GRAPH_MODE_CUB=="DEL":
        tails0, heads0, dists0, angs0 = GM_GenData._build_delaunay_graph(pts0)
        tails1, heads1, dists1, angs1 = GM_GenData._build_delaunay_graph(pts1)
    elif GM_GenData.GRAPH_MODE_CUB=="KNN":
        tails0, heads0, dists0, angs0 = GM_GenData._build_knn_graph(pts0,GM_GenData.NUM_K_WILLOW)
        tails1, heads1, dists1, angs1 = GM_GenData._build_knn_graph(pts1,GM_GenData.NUM_K_WILLOW)
    elif GM_GenData.GRAPH_MODE_CUB == "FULL":
        tails0, heads0, dists0, angs0 = GM_GenData._build_full_graph(pts0.shape[0])
        tails1, heads1, dists1, angs1 = GM_GenData._build_full_graph(pts1.shape[0])


    assignGraph = _gen_features_CUB_augmentation(pts0, tails0, heads0, descs0, patches0, pts1, tails1, heads1, descs1,
                                             patches1, use_train_set)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    image = {"category": category,
             "image1": matFiles[0],
             "image2": matFiles[1]}

    matchInfo["tails1"] = tails0.copy()
    matchInfo["heads1"] = heads0.copy()
    matchInfo["tails2"] = tails1.copy()
    matchInfo["heads2"] = heads1.copy()
    matchInfo["gidx1"] = gidx1.copy()
    matchInfo["gidx2"] = gidx2.copy()

    affinity=None

    return assignGraph, image, matchInfo, affinity


def _gen_features_CUB_augmentation(pts0, tails0, heads0,pts_fea0, patches0, pts1, tails1, heads1,pts_fea1, patches1, use_train_set):
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

            value = random.randint(0,100)
            if value > 70 and use_train_set:
                continue

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

def gen_random_graphs_CUB(rand,
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
            cid = rand.randint(0, 200)
        else:
            cid = category_id
        graph, image, matchInfo, affinity  = _gen_random_graph(rand, lib,
                                        CATEGORIES[cid],
                                        num_outlier_min_max=num_outlier_min_max,
                                        feaType = feaType,
                                        use_train_set = use_train_set)
        graphs.append(graph)
        images.append(image)
        matchInfos.append(matchInfo)
        affinities.append(affinity)

    return graphs, images, matchInfos, affinities




