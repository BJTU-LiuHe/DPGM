import numpy as np
import math
import cv2
import os
from scipy.spatial import Delaunay
from scipy import spatial
import torch
import re
from pathlib import Path

VISFEA_TYPE_RAWPIXEL = 'RawP'
VISFEA_TYPE_SIFT = 'SIFT'
VISFEA_TYPE_LFNET = 'LfNet'
VISFEA_TYPE_BBGM= "BBGM"
VISFEA_TYPE_PRETRAINED_LFNET = 'preLfNet'
VISFEA_TYPE_PRETRAINED_VGG16 = 'preVGG16'


"""### 整体参数 ###"""
DATASET="Pascal"            #当前数据集  Willow 或者 Pascal  Spair71k  CUB_2011  ICMPT
LEARNING_RATE=0.0008        #学习率
gpu_memory_fraction=1.0     #初始化时程序所占GPU内存的比例
GPU_ID="1"                  #用第几块GPU来训练模型

"""### QAPDATA ###"""
# ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
#['chr', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai']
#[ 'lipa', 'nug', 'scr', 'sko', 'ste', 'tai']
CLASS_QAPDATA = "sko"
LATENT_DIM_QAPDATA = 2
KEEPPROB_ENCODER_QAPDATA=1.0 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_QAPDATA=1.0 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_QAPDATA=1.0    #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_QAPDATA=1   #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_QAPDATA="DEL"     #构建graph的方式，KNN or DEL
NUM_K_QAPDATA=3              #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_QAPDATA=0.0000

"""### Willow库 参数 ###"""
LATENT_DIM_WILLOW=32          #隐含层中的维度
KEEPPROB_ENCODER_WILLOW=0.5 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_WILLOW=0.5 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_WILLOW=1.0   #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_WILLOW=16 #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_WILLOW="DEL"         #构建graph的方式，KNN or DEL
NUM_K_WILLOW=3                  #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_WILLOW=0.0000
"""### Pascal库 参数 (主要调整LATENT_DIM_PASCAL 和 MEAN_POOLING_INTERVAL) ###"""
LATENT_DIM_PASCAL=128       #隐含层中的维度
KEEPPROB_ENCODER_PASCAL=0.9 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_PASCAL=0.9 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_PASCAL=1.0    #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_PASCAL= 2   #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_PASCAL="DEL"     #构建graph的方式，KNN or DEL
NUM_K_PASCAL=3              #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_PASCAL=0.0000

"""### Spair71k 参数 (主要调整LATENT_DIM_PASCAL 和 MEAN_POOLING_INTERVAL) ###"""
LATENT_DIM_SPAIR71K=128       #隐含层中的维度
KEEPPROB_ENCODER_SPAIR71K=0.9 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_SPAIR71K=0.9 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_SPAIR71K=1.0    #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_SPAIR71K=2   #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_SPAIR71K="DEL"     #构建graph的方式，KNN or DEL
NUM_K_SPAIR71K=3              #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_SPAIR71K=0.0000

"""### CUB_2011 参数 (主要调整LATENT_DIM_PASCAL 和 MEAN_POOLING_INTERVAL) ###"""
LATENT_DIM_CUB=64       #隐含层中的维度
KEEPPROB_ENCODER_CUB=1.0 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_CUB=1.0 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_CUB=1.0    #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_CUB=8   #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_CUB="DEL"     #构建graph的方式，KNN or DEL
NUM_K_CUB=3              #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_CUB=0.0000

"""### ICMPT 参数 (主要调整LATENT_DIM_PASCAL 和 MEAN_POOLING_INTERVAL) ###"""
LATENT_DIM_ICMPT=64       #隐含层中的维度
KEEPPROB_ENCODER_ICMPT=1.0 #在encoder中的dropout中保留多少比例的节点
KEEPPROB_DECODER_ICMPT=1.0 #在decoder中的dropout中保留多少比例的节点
KEEPPROB_CONV_ICMPT=1.0    #在卷积层中的dropout中保留多少比例的节点
MEAN_POOLING_INTERVAL_ICMPT=8   #对vgg16视觉特征进行mean_pooling时每隔多少个元素做mean
GRAPH_MODE_ICMPT="DEL"     #构建graph的方式，KNN or DEL
NUM_K_ICMPT=3              #KNN 构图模式中，每个节点的邻居数量
REGULAR_RATE_ICMPT=0.0000


KEYPOINT_PATCH_WIDTH = 16
KEYPOINT_PATCH_HEIGHT = 16
# def _compute_SIFT_features(file, pts) :
#     # extract SIFT features
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     sift = cv2.xfeatures2d_SIFT.create()
#
#     kps = []
#     for i in range(pts.shape[0]):
#         kp = cv2.KeyPoint(pts[i][0], pts[i][1], 16, _class_id = 0)
#         kps.append(kp)
#
#     kps, descs = sift.compute(img, kps)
#     return kps, descs

def _map_cls_to_images(root, categories):
    cls_images = dict()
    for cls in categories:
        cls_images[cls] = os.listdir(os.path.join(root, cls))

    return cls_images

def _compute_SIFT_features(file, pts) :
    # extract SIFT features
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d_SIFT.create()

    kps = []
    desc = []
    for i in range(pts.shape[0]):
        kp = cv2.KeyPoint(pts[i][0], pts[i][1], 16, _class_id = 0)
        kps.append(kp)

    if len(kps) > 0:
        kps, desc = sift.compute(img, kps)
    else:
        desc = np.array(desc)
    return kps, desc

def _extract_img_patch(src, cor_x, cor_y, width, height):
    patch = np.zeros(shape = (height, width), dtype = src.dtype)

    img_h = src.shape[0]
    img_w = src.shape[1]

    cor_x = int(cor_x)
    cor_y = int(cor_y)
    if (cor_x < 0 or cor_x >= img_w or cor_y < 0 or cor_y >= img_h):
        return patch

    half_w = int(width / 2)
    half_h = int(height / 2)
    src_x = max(cor_x - half_w, 0)
    src_y = max(cor_y - half_h, 0)
    dst_x = max(half_w - cor_x, 0)
    dst_y = max(half_h - cor_y, 0)

    pad_x = max(0, half_w - cor_x)
    pad_x = max(pad_x, cor_x + half_w - img_w)
    pad_y = max(0, half_h - cor_y)
    pad_y = max(pad_y, cor_y + half_h - img_h)

    copy_x = width - pad_x
    copy_y = height - pad_y


    patch[dst_y:dst_y+copy_y, dst_x:dst_x+copy_x] = src[src_y:src_y+copy_y, src_x:src_x+copy_x]

    return patch

def _compute_image_patches(file, pts, width, height):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    patches = np.zeros(shape = (pts.shape[0], width, height), dtype = img.dtype)
    for i in range(pts.shape[0]):
        patch = _extract_img_patch(img, pts[i][0], pts[i][1], width, height)
        patches[i] = patch

    patches = patches.astype(np.float32) / 255.0

    return patches


def _build_threshold_graph(points, threshold) :
    distance = spatial.distance.cdist(points, points)

    idxs = np.where(distance <= threshold)

    dists = distance[idxs]
    tails = idxs[0]
    heads = idxs[1]

    edges = points[heads] - points[tails]
    angs = np.zeros(edges.shape[0], dtype = np.float)
    for i in range(edges.shape[0]):
        angs[i] = math.atan2(edges[i][1], edges[i][0])

    return tails, heads, dists, angs


def _build_delaunay_graph(points) :
    A = np.zeros(shape = (points.shape[0], points.shape[0]), dtype = np.float)  #N*N
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

def _build_knn_graph(points,num_k):
    dists,angs=None,None
    num_pts=points.shape[0]
    distance = spatial.distance.cdist(points, points)
    distance_sorted_index=np.argsort(distance,axis=1)
    knn_idx_=distance_sorted_index[:,1:1+num_k]
    knn_idx_=knn_idx_.reshape(-1)

    pts_idx=np.repeat(np.arange(0,num_pts),repeats=num_k)

    tails=pts_idx
    heads=knn_idx_

    return tails,heads, dists,angs

def _normalize_coordinates(points) :
    # normalize the coordinates
    deviation = np.nanstd(points, axis=0)
    points = np.transpose(points)
    points[0] = points[0] / deviation[0]
    points[1] = points[1] / deviation[1]
    points = np.transpose(points)
    return points

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

def _load_data(ROOT, file_name):

    dat_path = Path(os.path.join(ROOT, (file_name + '.dat')))
    sln_path = Path(os.path.join(ROOT, (file_name + '.sln')))
    dat_file = dat_path.open()
    sln_file = sln_path.open()

    def split_line(x):
        for _ in re.split(r'[,\s]', x.rstrip('\n')):
            if _ == "":
                continue
            else:
                yield int(_)

    dat_list = [[_ for _ in split_line(line)] for line in dat_file]
    sln_list = [[_ for _ in split_line(line)] for line in sln_file]

    prob_size = dat_list[0][0]

    # read data
    r = 0
    c = 0
    Fi = [[]]
    Fj = [[]]
    F = Fi
    for l in dat_list[1:]:
        F[r] += l
        c += len(l)
        assert c <= prob_size
        if c == prob_size:
            r += 1
            if r < prob_size:
                F.append([])
                c = 0
            else:
                F = Fj
                r = 0
                c = 0
    Fi = np.array(Fi, dtype=np.float32)
    Fj = np.array(Fj, dtype=np.float32)
    assert Fi.shape == Fj.shape == (prob_size, prob_size)

    # read solution
    sol = sln_list[0][1]
    perm_list = []
    for _ in sln_list[1:]:
        perm_list += _
    assert len(perm_list) == prob_size
    perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
    for r, c in enumerate(perm_list):
        perm_mat[r, c - 1] = 1

    Fi_t = torch.from_numpy(np.expand_dims(Fi, axis=0))
    Fj_t = torch.from_numpy(np.expand_dims(Fj, axis=0))

    aff_mat = _kronecker_torch(Fi_t, Fj_t)
    aff_mat = aff_mat.squeeze(0).numpy()
    # aff_mat = aff_mat.numpy()
    # perm_mat_t = torch.from_numpy(np.expand_dims(perm_mat, axis=0))
    # perm_mat_vec = perm_mat_t.transpose(1, 2).contiguous().view(1, -1, 1)
    # sol_v1 = torch.matmul(torch.matmul(perm_mat_vec.transpose(1, 2), aff_mat), perm_mat_vec).view(-1)

    adj1, adj2, gX, results = Fi, Fj, perm_mat, sol

    return adj1, adj2, gX, results, aff_mat