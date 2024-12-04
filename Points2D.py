import numpy as np

from scipy import spatial
from scipy.spatial import Delaunay
import math

from ctypes import *
import numpy.ctypeslib as npct


def _generate_2DPoints(rand,
             nInner = 10,
             nOutlier = 0,
             rho = 0.0,        # deformation range
             angle = 0.0,
             scale = 1.0):
    # parameters
    rgSize = 256 * np.sqrt((nInner + nOutlier) / 10)

    # generate inner points
    kps1 = {}
    kps2 = {}
    kps1["coords"] = rgSize * rand.uniform(size = (nInner, 2)) - (rgSize / 2)

#    kps1["weights"] = rand.exponential(1.0, size = nInner)
#   # white guassian noise for node weights
#    kps2["weights"] = kps1["weights"] + rand.uniform(-noise, noise, size=nInner)

    # deform for inner points coords by adding guassian noise
    deform = rho * rand.randn(nInner, 2)
    kps2["coords"] = kps1["coords"] + deform
    # scaling and rotation
    A = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    kps2["coords"] = np.transpose(np.dot(scale * A, np.transpose(kps2["coords"])))

    # generate outlier points in  both graph1 and graph2
    if nOutlier > 0 :
        coords1 = rgSize * rand.uniform(size = (nOutlier, 2)) - (rgSize / 2)
        coords2 = scale * (rgSize * rand.uniform(size = (nOutlier, 2)) - (rgSize / 2))
        kps1["coords"] = np.vstack((kps1["coords"], coords1))
        kps2["coords"] = np.vstack((kps2["coords"], coords2))

    # randomly re-order generated points
    pos_index1 = np.arange(0, nInner + nOutlier, 1)
    pos_index2 = np.arange(0, nInner + nOutlier, 1)
    rand.shuffle(pos_index1)
    rand.shuffle(pos_index2)
    kps1["coords"] = kps1["coords"][pos_index1]
    kps2["coords"] = kps2["coords"][pos_index2]

    # set ground-truth
    gX = np.zeros((nInner + nOutlier, nInner + nOutlier), np.bool)
    for i in range(nInner):
        gX[i][i] = True
    gX = np.transpose(np.transpose(gX[pos_index1])[pos_index2])

    return kps1, kps2, gX

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

def _gen_features_Points2D(pts0, tails0, heads0, pts1, tails1, heads1):
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

def _generate_graph_2DPoints(kps0, kps1, gX):
    tails0, heads0, dists0, angs0 = _build_delaunay_graph(kps0)
    tails1, heads1, dists1, angs1 = _build_delaunay_graph(kps1)
    assignGraph = _gen_features_Points2D(kps0, tails0, heads0, kps1, tails1, heads1)

    gidx1 = assignGraph["gidx1"]
    gidx2 = assignGraph["gidx2"]
    solutions = np.zeros(len(gidx1), np.bool)
    for i in range(len(gidx1)) :
        if gX[gidx1[i]][gidx2[i]] :
            solutions[i] = True
    assignGraph["solutions"] = solutions

    return assignGraph

def _generate_affinity_2DPoints(lib, kps1, kps2, gX) :
    n1 = kps1["coords"].shape[0]
    n2 = kps2["coords"].shape[0]

    distance1 = spatial.distance.cdist(kps1["coords"], kps1["coords"])
    distance2 = spatial.distance.cdist(kps2["coords"], kps2["coords"])

    # generate group and ground-truth label
    gidx1 = []
    gidx2 = []
    solutions = []
    index = 0
    for i2 in range(n2):
        for i1 in range(n1):
            gidx1.append(i1)
            gidx2.append(i2)
            solutions.append(gX[i1, i2])
            index = index + 1

    num_matches = index

    dist1 = np.float32(np.reshape(distance1, n1 * n1, order = 'C'))
    dist2 = np.float32(np.reshape(distance2, n2 * n2, order = 'C'))
    g1 = np.array(gidx1)
    g2 = np.array(gidx2)
    K = np.zeros(num_matches * num_matches, np.float32)
    count = lib.build_affinity_2DPoints(n1, n2, dist1, dist2, num_matches, g1, g2,  K)
    K = np.float64(np.reshape(K, (num_matches, num_matches), order = 'C'))

    return K,gidx1, gidx2, solutions


def _gen_random_graph(rand,
                         nInner,
                         nOutlier,
                         rho,
                         angle=0.0,
                         scale=1.0):
    """Creates a geographic threshold graph.

    Args:
      rand: A random seed for the graph generator. Default= None.
      num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
      dimensions: (optional) An `int` number of dimensions for the positions.
        Default= 2.
      theta: (optional) A `float` threshold parameters for the geographic
        threshold graph's threshold. Large values (1000+) make mostly trees. Try
        20-60 for good non-trees. Default=1000.0.
      rate: (optional) A rate parameter for the node weight exponential sampling
        distribution. Default= 1.0.

    Returns:
      The graph.
    """

    # generate 2D points
    kps1, kps2, gX = _generate_2DPoints(rand, nInner, nOutlier,
                                      rho, angle, scale)

    assignGraph = _generate_graph_2DPoints(kps1["coords"], kps2["coords"], gX)

    return assignGraph



def gen_random_graphs_Points2D(rand,
                             num_examples,
                             num_inner_min_max,
                             num_outlier_min_max):

    # lib = npct.load_library("GMBuilder.dll", ".")  # 引入动态链接库，load_library的用法可参考官网文档
    # lib.build_affinity_2DPoints.argtypes = [c_int,
    #                                         c_int,
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    #                                         c_int,
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
    #                                         npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]


    graphs = []
    for _ in range(num_examples):
        nInner = rand.randint(*num_inner_min_max)
        nOutlier = rand.randint(*num_outlier_min_max)
        rho = rand.randint(0, 21)
     #   rho = 5

        graph = _gen_random_graph(rand, nInner, nOutlier, rho)
        graphs.append(graph)


    return graphs



