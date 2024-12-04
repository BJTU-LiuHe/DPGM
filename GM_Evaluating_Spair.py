import tensorflow as tf
import numpy as np
import scipy.io
import time
import torch
from ctypes import *
import numpy.ctypeslib as npct
import cv2

from graph_nets.demos import models
import GM_Core_featured as gmc
import GM_GenData

import Points2D
import CMUHouse as CMU
import Willow
import PascalVoc as VOC
import Spair71k_test as Spair71k
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
NODE_VISFEA_TYPE = GM_GenData.VISFEA_TYPE_PRETRAINED_VGG16


def gen_test_data_Points2D(rand, num_sample_per_category):

    lib = npct.load_library("GMBuilder.dll", ".")  # 引入动态链接库，load_library的用法可参考官网文档
    lib.build_affinity_2DPoints.argtypes = [c_int,
                                            c_int,
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            c_int,
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]

    nInner = 10
    rho = 5
    for nOutlier in range(11):
        for i in range(num_sample_per_category):
            affinity = Points2D._gen_random_affinity(rand, lib, nInner, nOutlier, rho)
            matFile = "test_data/Points2D/outliers{:02d}_{:05d}.mat".format(nOutlier, i)
            scipy.io.savemat(matFile, affinity)

    nOutlier = 5
    for rho in range(0, 11, 1):
        for i in range(num_sample_per_category):
            affinity = Points2D._gen_random_affinity(rand, lib, nInner, nOutlier, rho)
            matFile = "test_data/Points2D/noise{:02d}_{:05d}.mat".format(rho, i)
            scipy.io.savemat(matFile, affinity)


def gen_test_data_CMU(rand,
                      num_inner_min_max):

    num_frames, num_points, XTs = CMU._load_data_from_mat("cmuHouse.mat")

    lib = npct.load_library("GMBuilder.dll",".")  #引入动态链接库，load_library的用法可参考官网文档
    lib.build_affinity_CMUHouse.argtypes = [c_int,
                                            c_int,
                                            npct.ndpointer(dtype = np.int, ndim = 1, flags = "C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            c_int,
                                            c_int,
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]

    lib.build_affinity_CMUHouse2.argtypes = [c_int,
                                            c_int,
                                            npct.ndpointer(dtype = np.float32, ndim = 1, flags = "C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
                                            c_int,
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
                                            npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")]

    max_frames = 111
    for gaps in range(10, 101, 10):
        for start_frame in range(0, max_frames - gaps, 1):
            frame_indexs = (start_frame, start_frame + gaps)
            affinity = CMU._gen_random_affinity(rand,
                                            lib,
                                            XTs,
                                            frame_indexs,
                                            num_inner_min_max=num_inner_min_max)

            matFile = "test_data/CMUHouse/gaps{:03d}_{:03d}.mat".format(gaps, start_frame)
            scipy.io.savemat(matFile, affinity)



def gen_test_data_Willow(rand,
                         num_sample_per_category,
                         num_inner_min_max,
                         num_outlier_min_max):
    num_categories = 5
    batch_size = 100
    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        for j in range(num_batch):
            affinities, images = Willow.gen_random_affinities_Willow(rand,
                                                                     batch_size,
                                                                     num_inner_min_max,
                                                                     num_outlier_min_max,
                                                                     category_id = i)
            for k in range(batch_size):
                matFile = "test_data/Willow/{}_{:05d}.mat".format(images[k]["category"], j * batch_size + k)
                D = affinities[k].copy()
                D.update(images[k])
                scipy.io.savemat(matFile, D)
                print(matFile)

def gen_test_data_VOC(rand,
                      num_sample_per_category,
                      num_inner_min_max,
                      num_outlier_min_max):
    num_categories = 20
    batch_size = 100
    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        for j in range(num_batch):
            affinities, images = VOC.gen_random_affinities_VOC(rand,
                                                                     batch_size,
                                                                     num_inner_min_max,
                                                                     num_outlier_min_max,
                                                                     category_id = i)
            for k in range(batch_size):
                matFile = "test_data/VOC/{}_{:05d}.mat".format(images[k]["category"], j * batch_size + k)
                D = affinities[k].copy()
                D.update(images[k])
                scipy.io.savemat(matFile, D)
                print(matFile)


def read_from_mat(mat_file) :
  data = scipy.io.loadmat(mat_file)  # 读取mat文件
  # print(data.keys())   # 查看mat文件中的所有变量
  K = data['K']
  group1 = data['g1']
  group2 = data['g2']
  gt = (data['gt'])

  solutions = np.zeros(gt.shape[0], np.bool)
  for i in range(gt.shape[0]) :
    if gt[i] > 0 :
      solutions[i] = True

  return K, group1, group2, solutions


def load_trained_model(model_file):
    tf.reset_default_graph()

    keepprob_encoder_ph,keepprob_decoder_ph=tf.placeholder(dtype=tf.float64,shape=None),tf.placeholder(dtype=tf.float64,shape=None)
    keep_prob_conv_ph=tf.placeholder(dtype=tf.float64,shape=None)
    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 3
    num_processing_steps_ge = 3

    # Data.
    # Input and target placeholders.
    batch_size_ge = 10
    num_inner_min_max = (10, 11)
    num_outlier_min_max = (0, 1)

    seed = 0
    rand = np.random.RandomState(seed=seed)

    # Data.
    gmc.NODE_OUTPUT_SIZE = 1
    # Input and target placeholders.
    input_ph, target_ph, loss_cof_ph = gmc.create_placeholders(
        rand, batch_size_ge, num_inner_min_max, num_outlier_min_max, visfeaType = NODE_VISFEA_TYPE,dataset=GM_GenData.DATASET)
    # Instantiate the model.
    # model = models.EncodeProcessDecode(node_input_size = 514, edge_output_size=1, node_output_size=gmc.NODE_OUTPUT_SIZE, group_output_size=1)
    model = models.EncodeProcessDecode(visfeaType = NODE_VISFEA_TYPE,
                                       dynamic_core_num= 0,
                                       edge_output_size=1,
                                       node_output_size=gmc.NODE_OUTPUT_SIZE,
                                       group_output_size = 1)
    # A list of outputs, one per processing step.
#    output_ops_tr = model(input_ph, num_processing_steps_tr)
    output_ops_ge = model(input_ph, num_processing_steps_ge,keepprob_encoder_ph,keepprob_decoder_ph,keep_prob_conv_ph)

    # This cell resets the Tensorflow session, but keeps the same computational graph.
    try:
        sess.close()
    except NameError:
        pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # load trained model
    saver = tf.train.Saver(max_to_keep = 1)
    saver.restore(sess, model_file)

    return sess, input_ph, target_ph, loss_cof_ph, keepprob_encoder_ph,keepprob_decoder_ph, keep_prob_conv_ph,output_ops_ge

def combineImages(imgPath1, imgPath2):
    # combine images
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)

    if img1.shape[0] > img2.shape[0]:
        black = np.zeros(shape=(img1.shape[0] - img2.shape[0], img2.shape[1], img2.shape[2]), dtype = img2.dtype)
        img2 = np.vstack([img2, black])
    elif img1.shape[0] < img2.shape[0]:
        black = np.zeros(shape=(img2.shape[0] - img1.shape[0], img1.shape[1], img1.shape[2]), dtype = img1.dtype)
        img1 = np.vstack([img1, black])
    img = np.hstack([img1, img2])
    return img

def drawMatches(imgPath1, pts1, tails1, heads1, gidx1, imgPath2, pts2, tails2, heads2, gidx2, X, gX):
    # combine images
    img1 = cv2.imread(imgPath1)
    img2 = cv2.imread(imgPath2)

    if img1.shape[0] > img2.shape[0]:
        black = np.zeros(shape=(img1.shape[0] - img2.shape[0], img2.shape[1], img2.shape[2]), dtype = img2.dtype)
        img2 = np.vstack([img2, black])
    elif img1.shape[0] < img2.shape[0]:
        black = np.zeros(shape=(img2.shape[0] - img1.shape[0], img1.shape[1], img1.shape[2]), dtype = img1.dtype)
        img1 = np.vstack([img1, black])
    img = np.hstack([img1, img2])


    # draw two graphs
    color = (0, 255, 255)
    for i in range(len(tails1)):
        ptt = pts1[tails1[i]].astype(np.int)
        pth = pts1[heads1[i]].astype(np.int)
        cv2.line(img, (ptt[0], ptt[1]), (pth[0], pth[1]), color, 1, cv2.LINE_AA)
    for i in range(len(tails2)):
        ptt = pts2[tails2[i]].astype(np.int)
        pth = pts2[heads2[i]].astype(np.int)
        ptt[0] = ptt[0] + img1.shape[1]
        pth[0] = pth[0] + img1.shape[1]
        cv2.line(img, (ptt[0], ptt[1]), (pth[0], pth[1]), color, 1, cv2.LINE_AA)

    # draw matches
    color_g = (0, 255, 0)
    color_r = (0, 0, 255)
    for i in range(len(X)):
        if X[i] > 0:
            ptm1 = pts1[gidx1[i]].astype(np.int)
            ptm2 = pts2[gidx2[i]].astype(np.int)
            ptm2[0] = ptm2[0] + img1.shape[1]

            if X[i] == gX[i]:
                color = color_g
            else:
                color = color_r

            cv2.line(img, (ptm1[0], ptm1[1]), (ptm2[0], ptm2[1]), color, 2, cv2.LINE_AA)

    #cv2.imshow("matches", img)
    #cv2.waitKey(0)

    return img

def evaluate_Points2D(sess,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      output_ops_ge):

    num_sample_per_category = 1000
    batch_size = 100
    accuracy_outliers = []

    rho = 20
    for nOutlier in range(0, 11, 1):
        num_batch = int(num_sample_per_category / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs = []
            for k in range(batch_size):
                graph = Points2D._gen_random_graph(rand,
                                                   nInner = 10,
                                                   nOutlier = nOutlier,
                                                   rho = rho)
                graphs.append(graph)

            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                  input_ph,
                                                                  target_ph,
                                                                  loss_cof_ph)
            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)

            print("{}, {}, {:.4f}".format(nOutlier, j, correct_gt_ge))

            accuracy = accuracy + correct_gt_ge

        accuracy = accuracy / num_batch
        accuracy_outliers.append(accuracy)


    # nOutlier = 5
    # accuracy_rhos = []
    # for rho in range(0, 21, 2):
    #     num_batch = int(num_sample_per_category / batch_size)
    #     accuracy = 0.0
    #     for j in range(num_batch):
    #         graphs = []
    #         for k in range(batch_size):
    #             graph = Points2D._gen_random_graph(rand,
    #                                                nInner = 10,
    #                                                nOutlier = nOutlier,
    #                                                rho = rho)
    #             graphs.append(graph)
    #
    #         feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
    #                                                               input_ph,
    #                                                               target_ph,
    #                                                               loss_cof_ph)
    #         t0 = time.time()
    #         eval_values = sess.run({
    #             "target": target_ph,
    #             "outputs": output_ops_ge},
    #             feed_dict=feed_dict)
    #         eval_time = time.time() - t0
    #
    #         nodes = eval_values["outputs"][-1].nodes.copy()
    #         group_indices = eval_values["outputs"][-1].group_indices_1.copy()
    #         x, count = gmc.greedy_mapping(nodes, group_indices)
    #         correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
    #             eval_values["target"], eval_values["outputs"][-1], use_edges=False)
    #
    #         print("{}, {}, {:.4f}".format(rho, j, correct_gt_ge))
    #
    #         accuracy = accuracy + correct_gt_ge
    #
    #     accuracy = accuracy / num_batch
    #     accuracy_rhos.append(accuracy)


    # for i in range(len(accuracy_rhos)):
    #     print("RHO = {}: acc = {:.4f}".format(2 * i, accuracy_rhos[i]))
    for i in range(len(accuracy_outliers)):
        print("Outlier = {}: acc = {:.4f}".format(i, accuracy_outliers[i]))

def evaluate_CMUHouse(sess,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      output_ops_ge):
    mat_file = "cmuHouse.mat"
    num_frames, num_points, XTs = CMU._load_data_from_mat(mat_file)

    max_frames = 111
    # for gaps in range(10, 101, 10):
    #     graphs = []
    #     for start_frame in range(0, max_frames - gaps, 1):
    #         graph = CMU._gen_random_graph(rand,
    #                                       XTs,
    #                                       frame_indexs = (start_frame, start_frame + gaps),
    #                                       num_inner_min_max = (30, 31))
    #         graphs.append(graph)
    #
    #     feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
    #                                                           input_ph,
    #                                                           target_ph,
    #                                                           loss_cof_ph)
    #     t0 = time.time()
    #     eval_values = sess.run({
    #         "target": target_ph,
    #         "outputs": output_ops_ge},
    #         feed_dict=feed_dict)
    #     eval_time = time.time() - t0
    #
    #     nodes = eval_values["outputs"][-1].nodes.copy()
    #     group_indices = eval_values["outputs"][-1].group_indices_1.copy()
    #     x, count = gmc.greedy_mapping(nodes, group_indices)
    #     correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
    #         eval_values["target"], eval_values["outputs"][-1], use_edges=False)
    #
    #     print("gaps = {}: {:.4f}".format(gaps, correct_gt_ge))

    gaps = 50
    for inners in range(10, 31, 2):
        graphs = []
        for start_frame in range(0, max_frames - gaps, 1):
            graph = CMU._gen_random_graph(rand,
                                          XTs,
                                          frame_indexs = (start_frame, start_frame + gaps),
                                          num_inner_min_max = (inners, inners + 1))
            graphs.append(graph)

        feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                              input_ph,
                                                              target_ph,
                                                              loss_cof_ph)
        t0 = time.time()
        eval_values = sess.run({
            "target": target_ph,
            "outputs": output_ops_ge},
            feed_dict=feed_dict)
        eval_time = time.time() - t0

        nodes = eval_values["outputs"][-1].nodes.copy()
        group_indices = eval_values["outputs"][-1].group_indices_1.copy()
        x, count = gmc.greedy_mapping(nodes, group_indices)
        correct_gt_ge, correct_all_ge, solved_ge, matches_ge = gmc.compute_accuracy(
            eval_values["target"], eval_values["outputs"][-1], use_edges=False)

        print("inners = {}: {:.4f}".format(inners, correct_gt_ge))


def evaluate_Willow(sess,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      keepprob_encoder_ph,
                      keepprob_decoder_ph,
                      kepp_prob_conv_ph,
                      output_ops_ge):

    WILLOW_CATEGORIES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

    num_categories = 5
    num_sample_per_category = 100
    batch_size = 1
    keepprob_encoder, keepprob_decoder = 1.0, 1.0
    keepprob_conv=1.0
    accuracy_categories = []

    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs, images, matchInfos, affinities = Willow.gen_random_graphs_Willow(rand,
                                                  batch_size,
                                                  num_inner_min_max,
                                                  num_outlier_min_max,
                                                  feaType=NODE_VISFEA_TYPE,
                                                  use_train_set=False,
                                                  category_id=i)
            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                  input_ph,
                                                                  target_ph,
                                                                  loss_cof_ph)


            feed_dict[keepprob_encoder_ph]=keepprob_encoder
            feed_dict[keepprob_decoder_ph]=keepprob_decoder
            feed_dict[kepp_prob_conv_ph]=keepprob_conv
            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)
            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            # correct_gt_ge, correct_all_ge, solved_ge, matches_ge, Xs, gXs = gmc.compute_accuracy(
            #     eval_values["target"], eval_values["outputs"][-1], use_edges=False)
            correct_all_ge, correct_gt_ge, matches_ge, Xs, gXs = gmc.compute_accuracy_hungurian(
                eval_values["target"], eval_values["outputs"][-1])

            print("{}, {}, {:.4f}".format(i, j, correct_gt_ge))

            accuracy = accuracy + correct_gt_ge

            num_matches = np.sum(gXs[0])
            num_corrects = np.sum(Xs[0] * gXs[0])

            """
            # draw match results
            img = drawMatches(images[0]["image1"], matchInfos[0]["pts1"], matchInfos[0]["tails1"], matchInfos[0]["heads1"], matchInfos[0]["gidx1"],
                        images[0]["image2"], matchInfos[0]["pts2"], matchInfos[0]["tails2"], matchInfos[0]["heads2"], matchInfos[0]["gidx2"],
                        Xs[0], gXs[0])
            saveImgPath = "test_data/Willow/{}_{:05d}_LGM_{}_{}.jpg".format(images[0]["category"], j, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)
            """
            # save affinities for learning-free algorithms
            # matFile = "test_data/Willow/{}_{:05d}.mat".format(images[0]["category"], j)
            # D = {"gX": gXs[0]}
            # D.update(affinities[0])
            # D.update(images[0])
            # D.update(matchInfos[0])
            # scipy.io.savemat(matFile, D)
            # print(matFile)


        accuracy = accuracy / num_batch
        accuracy_categories.append(accuracy)

    avg_accuracy = 0.0
    for i in range(num_categories):
        avg_accuracy = avg_accuracy + accuracy_categories[i]
        print("{}: {:.4f}".format(WILLOW_CATEGORIES[i], accuracy_categories[i]))

    avg_accuracy = avg_accuracy / num_categories
    print("AVG_ACCURACY: {:.4f}".format(avg_accuracy))


def evaluate_VOC(sess,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      keepprob_encoder_ph,
                      keepprob_decoder_ph,
                      kepp_prob_conv_ph,
                      output_ops_ge):
    VOC_CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                      "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                      "tvmonitor"]

    num_categories = 20
    num_sample_per_category = 1000
    batch_size = 1
    accuracy_categories = []

    keepprob_encoder, keepprob_decoder = 1.0, 1.0
    keepprob_conv = 1.0

    for i in range(num_categories):
        num_batch = int(num_sample_per_category / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs, images, matchInfos, affinities = VOC.gen_random_graphs_VOC(rand,
                                                  batch_size,
                                                  num_inner_min_max,
                                                  num_outlier_min_max,
                                                  feaType = NODE_VISFEA_TYPE,
                                                  use_train_set = False,
                                                  category_id = i)
            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                      input_ph,
                                                                      target_ph,
                                                                      loss_cof_ph)

            feed_dict[keepprob_encoder_ph] = keepprob_encoder
            feed_dict[keepprob_decoder_ph] = keepprob_decoder
            feed_dict[kepp_prob_conv_ph] = keepprob_conv

            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)

            eval_time = time.time() - t0

            nodes = eval_values["outputs"][-1].nodes.copy()
            group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            x, count = gmc.greedy_mapping(nodes, group_indices)
            #correct_gt_ge, correct_all_ge, solved_ge, matches_ge, Xs, gXs = gmc.compute_accuracy(
            #    eval_values["target"], eval_values["outputs"][-1], use_edges=False)
            correct_all_ge, correct_gt_ge, matches_ge, Xs, gXs = gmc.compute_accuracy_hungurian(
                eval_values["target"], eval_values["outputs"][-1])
            print("{}, {}, {:.4f}".format(i, j, correct_gt_ge))

            accuracy = accuracy + correct_gt_ge

            num_matches = np.sum(gXs[0])
            num_corrects = np.sum(Xs[0] * gXs[0])
            """
            # draw match results
            img = drawMatches(images[0]["image1"], matchInfos[0]["pts1"], matchInfos[0]["tails1"], matchInfos[0]["heads1"], matchInfos[0]["gidx1"],
                        images[0]["image2"], matchInfos[0]["pts2"], matchInfos[0]["tails2"], matchInfos[0]["heads2"], matchInfos[0]["gidx2"],
                        Xs[0], gXs[0])
            saveImgPath = "test_data/VOC/{}_{:05d}_LGM_{}_{}.jpg".format(images[0]["category"], j, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)
            """
            # save affinities for learning-free algorithms
            # matFile = "test_data/VOC/{}_{:05d}.mat".format(images[0]["category"], j)
            # D = {"gX": gXs[0]}
            # D.update(affinities[0])
            # D.update(images[0])
            # D.update(matchInfos[0])
            # scipy.io.savemat(matFile, D)
            # print(matFile)


        accuracy = accuracy / num_batch
        accuracy_categories.append(accuracy)

    avg_accuracy = 0.0
    for i in range(num_categories):
        avg_accuracy = avg_accuracy + accuracy_categories[i]
        print("{}: {:.4f}".format(VOC_CATEGORIES[i], accuracy_categories[i]))

    avg_accuracy = avg_accuracy / num_categories
    print("AVG_ACCURACY: {:.4f}".format(avg_accuracy))

def evaluate_SPair(sess,
                      input_ph,
                      target_ph,
                      loss_cof_ph,
                      keepprob_encoder_ph,
                      keepprob_decoder_ph,
                      kepp_prob_conv_ph,
                      output_ops_ge):
    # SPair_CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    #               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
    SPair_CATEGORIES = ['aeroplane']
    save_root = "save_results"

    test_dataset_dict=Spair71k.test_dataset_dict
    nums_samples = list()
    for SPair_CATEGORY in SPair_CATEGORIES:
        nums_samples.append(len(test_dataset_dict[SPair_CATEGORY]))
    num_categories = len(SPair_CATEGORIES)
    batch_size = 1
    accuracy_categories = []

    keepprob_encoder, keepprob_decoder = 1.0, 1.0
    keepprob_conv = 1.0

    for i in range(num_categories):
        save_path = os.path.join(save_root, SPair_CATEGORIES[i])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        num_batch = int(nums_samples[i] / batch_size)
        accuracy = 0.0
        for j in range(num_batch):
            graphs, images, matchInfos, affinities = Spair71k.gen_random_graphs_Spair17k(rand,
                                                  batch_size,
                                                  num_inner_min_max,
                                                  num_outlier_min_max,
                                                  feaType = NODE_VISFEA_TYPE,
                                                  use_train_set = False,
                                                  category_id = i,
                                                    sample_idx= j)

            feed_dict, raw_graph = gmc.create_feed_dict_by_graphs(graphs,
                                                                      input_ph,
                                                                      target_ph,
                                                                      loss_cof_ph)

            # feed_dict, raw_graphs = gmc.create_feed_dict(
            #     rand, batch_size, num_inner_min_max, num_outlier_min_max,
            #     NODE_VISFEA_TYPE, input_ph, target_ph, loss_cof_ph,
            #     use_train_set=False, dataset=GM_GenData.DATASET, category=i)

            feed_dict[keepprob_encoder_ph] = keepprob_encoder
            feed_dict[keepprob_decoder_ph] = keepprob_decoder
            feed_dict[kepp_prob_conv_ph] = keepprob_conv

            t0 = time.time()
            eval_values = sess.run({
                "target": target_ph,
                "outputs": output_ops_ge},
                feed_dict=feed_dict)

            eval_time = time.time() - t0

            correct_gt_ge, correct_all_ge, solved_ge, matches_ge, Xs, gXs = gmc.compute_accuracy(
                eval_values["target"], eval_values["outputs"][-1], use_edges=False)
            #correct_all_ge, correct_gt_ge, matches_ge, Xs, gXs = gmc.compute_accuracy_hungurian(
            #    eval_values["target"], eval_values["outputs"][-1])

            print("{}, {}, {:.4f}".format(i, j, correct_gt_ge))

            # nodes = eval_values["outputs"][-1].nodes.copy()
            # nodes_gt = eval_values["target"].nodes.copy()
            # group_indices = eval_values["outputs"][-1].group_indices_1.copy()
            # x, count = gmc.greedy_mapping(nodes, group_indices)
            # num_nodes = matchInfos["pts1"].shape[0]
            # x = x.reshape((num_nodes, num_nodes))
            # x_gt = nodes_gt.reshape((num_nodes, num_nodes))
            # image_name1, image_name2 = matchInfos["image_pair"]
            # matchInfos["perm"] = x
            # matchInfos["perm_gt"] = x_gt
            # matchInfos["accuracy"] = correct_gt_ge
            # save_file = os.path.join(save_path, image_name1 + "-" + image_name2 + ".pth")
            # torch.save(matchInfos, save_file)

            accuracy = accuracy + correct_gt_ge

            num_matches = np.sum(gXs[0])
            num_corrects = np.sum(Xs[0] * gXs[0])
            """
            # draw match results
            img = drawMatches(images[0]["image1"], matchInfos[0]["pts1"], matchInfos[0]["tails1"], matchInfos[0]["heads1"], matchInfos[0]["gidx1"],
                        images[0]["image2"], matchInfos[0]["pts2"], matchInfos[0]["tails2"], matchInfos[0]["heads2"], matchInfos[0]["gidx2"],
                        Xs[0], gXs[0])
            saveImgPath = "test_data/VOC/{}_{:05d}_LGM_{}_{}.jpg".format(images[0]["category"], j, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)
            """
            # save affinities for learning-free algorithms
            # matFile = "test_data/VOC/{}_{:05d}.mat".format(images[0]["category"], j)
            # D = {"gX": gXs[0]}
            # D.update(affinities[0])
            # D.update(images[0])
            # D.update(matchInfos[0])
            # scipy.io.savemat(matFile, D)
            # print(matFile)


        accuracy = accuracy / num_batch
        accuracy_categories.append(accuracy)

    avg_accuracy = 0.0
    for i in range(num_categories):
        avg_accuracy = avg_accuracy + accuracy_categories[i]
        print("{}: {:.4f}".format(SPair_CATEGORIES[i], accuracy_categories[i]))

    avg_accuracy = avg_accuracy / num_categories
    print("AVG_ACCURACY: {:.4f}".format(avg_accuracy))

def drawWillowMatches():
    WILLOW_CATEGORIES = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

    num_categories = 5
    num_sample_per_category = 100

    for i in range(num_categories):
        for k in range(num_sample_per_category):
            matFile = "test_data/Willow/{}_{:05d}.mat".format(WILLOW_CATEGORIES[i], k)
            matchFile = "save/Willow/{}_{:05d}_MatchX.mat".format(WILLOW_CATEGORIES[i], k)
            data = scipy.io.loadmat(matFile)
            matches = scipy.io.loadmat(matchFile)

            img1 = data["image1"][0]
            img2 = data["image2"][0]
            pts1 = data["pts1"]
            pts2 = data["pts2"]
            tails1 = data["tails1"][0]
            tails2 = data["tails2"][0]
            heads1 = data["heads1"][0]
            heads2 = data["heads2"][0]
            gidx1 = data["gidx1"][0]
            gidx2 = data["gidx2"][0]
            gX = data["gX"]

            srcImg = combineImages(img1, img2)
            saveImgPath = "test_data/Willow/{}_{:05d}_src.jpg".format(WILLOW_CATEGORIES[i], k)
            cv2.imwrite(saveImgPath, srcImg)

            continue

            num_matches = np.sum(gX)

            # IPFP_S
            X = matches["X_IPFP_S"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_IPFPS_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # IPFP_U
            X = matches["X_IPFP_U"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_IPFPU_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # GAGM
            X = matches["X_GAGM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_GAGM_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # RRWM
            X = matches["X_RRWM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_RRWM_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # PSM
            X = matches["X_PSM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_PSM_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # GNCCP
            X = matches["X_GNCCP"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_GNCCP_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # ABPF_G
            X = matches["X_ABPF_G"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/Willow/{}_{:05d}_ABPF_{}_{}.jpg".format(WILLOW_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)


def drawPascalMatches():
    VOC_CATEGORIES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                      "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                      "tvmonitor"]
    num_categories = 20
    num_sample_per_category = 100

#    for i in range(num_categories):
    for i in [17]:
#        for k in range(num_sample_per_category):
        for k in [94]:
            matFile = "test_data/VOC/{}_{:05d}.mat".format(VOC_CATEGORIES[i], k)
            matchFile = "save/VOC/{}_{:05d}_MatchX.mat".format(VOC_CATEGORIES[i], k)
            data = scipy.io.loadmat(matFile)
            matches = scipy.io.loadmat(matchFile)

            img1 = data["image1"][0]
            img2 = data["image2"][0]
            pts1 = data["pts1"]
            pts2 = data["pts2"]
            tails1 = data["tails1"][0]
            tails2 = data["tails2"][0]
            heads1 = data["heads1"][0]
            heads2 = data["heads2"][0]
            gidx1 = data["gidx1"][0]
            gidx2 = data["gidx2"][0]
            gX = data["gX"]

            srcImg = combineImages(img1, img2)
            saveImgPath = "test_data/VOC/{}_{:05d}_src.jpg".format(VOC_CATEGORIES[i], k)
            cv2.imwrite(saveImgPath, srcImg)

            num_matches = np.sum(gX)

            #LGM
            X = gX.copy()
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_LGM_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            continue

            # IPFP_S
            X = matches["X_IPFP_S"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_IPFPS_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)


            # IPFP_U
            X = matches["X_IPFP_U"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_IPFPU_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # GAGM
            X = matches["X_GAGM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_GAGM_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # RRWM
            X = matches["X_RRWM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_RRWM_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # PSM
            X = matches["X_PSM"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_PSM_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # GNCCP
            X = matches["X_GNCCP"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_GNCCP_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)

            # ABPF_G
            X = matches["X_ABPF_G"]
            num_corrects = np.sum(X * gX)
            img = drawMatches(img1, pts1, tails1, heads1, gidx1, img2, pts2, tails2, heads2, gidx2, X, gX)
            saveImgPath = "test_data/VOC/{}_{:05d}_ABPF_{}_{}.jpg".format(VOC_CATEGORIES[i], k, num_corrects, num_matches)
            cv2.imwrite(saveImgPath, img)


# drawWillowMatches()
# drawPascalMatches()

## ***********************************************************************
## test data generation
## ************************************************************************

num_inner_min_max = (10, 11)
num_outlier_min_max = (0, 1)
#
seed = 0
rand = np.random.RandomState(seed=seed)
#
#gen_test_data_Willow(rand, 1000, num_inner_min_max, num_outlier_min_max)
#gen_test_data_VOC(rand, 100, num_inner_min_max, num_outlier_min_max)
#gen_test_data_CMU(rand, num_inner_min_max)
#gen_test_data_Points2D(rand, 500)


## ***********************************************************************
## evaluation
## **********************************************************************

# model_file = "trained_models/GM_2DPoints-22020"
#model_file = "trained_models/GM_CMUHouse-11300"
# model_file = "trained_models/Willow_RawP-lr=0.0005-6650"
# model_file = "trained_models/truncation/new_sinkhorn_pos_w1_Spair71k_EKpb-0.8_DKpb-1.0_CKpb-0.8_MPI-8_LD-64_LR-0.002_GM-DEL_RR-0.0-BBGM_PB-5/best_model_acc-9000"
# model_file = "trained_models_debug/Spair71k_EKpb-1.0_DKpb-1.0_CKpb-1.0_MPI-2_LD-128_LR-0.0003_GM-DEL-preVGG16/best_model_acc-20550"
model_file = "trained_models_R1_augmentation/best_Spair71k_EKpb-1.0_DKpb-1.0_CKpb-1.0_MPI-2_LD-128_LR-0.0008_GM-DEL_RR-0.0-BBGM_PB-5_iter_3_50000/Spair71k_RELOAD/Spair71k_RELOAD/best_model_acc-350000"
sess, input_ph, target_ph, loss_cof_ph, keep_prob_encoder_ph,keep_prob_decoder_ph,kepp_prob_conv_ph,output_ops_ge= load_trained_model(model_file)
# evaluate_Points2D(sess, input_ph, target_ph, loss_cof_ph, output_ops_ge)
#evaluate_CMUHouse(sess, input_ph, target_ph, loss_cof_ph, output_ops_ge)
# evaluate_Willow(sess, input_ph, target_ph, loss_cof_ph, keep_prob_encoder_ph,keep_prob_decoder_ph,kepp_prob_conv_ph,output_ops_ge)
# evaluate_VOC(sess, input_ph, target_ph, loss_cof_ph, keep_prob_encoder_ph,keep_prob_decoder_ph,kepp_prob_conv_ph,output_ops_ge)
evaluate_SPair(sess, input_ph, target_ph, loss_cof_ph, keep_prob_encoder_ph,keep_prob_decoder_ph,kepp_prob_conv_ph,output_ops_ge)