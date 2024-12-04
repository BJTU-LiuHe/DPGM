from graph_nets.demos import models
from graph_nets import utils_featured_graph

import time
#import threading
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import GM_Core_featured as gmc
import GM_GenData
import qaplib
import random
import torch
import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = GM_GenData.GPU_ID

categories = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
data_list = qaplib.data_list
# data_dict = qaplib.data_dict
baseline_file = "/home/6T/lh/data/graph_matching/QAP_baseline.txt"

baseline_res_dict = {}

fid = open(baseline_file, "r")
line = fid.readline()
while not line == "":
    name, value = line.split()
    if name in data_list:
        baseline_res_dict[name] = float(value)
    line = fid.readline()
fid.close()

sample_index = list(range(len(data_list)))


TRAIN_DATASET = GM_GenData.DATASET
LEARNING_RATE= 0.0008
CATEGORY = GM_GenData.CLASS_QAPDATA
KEEPPROB_ENCODER=GM_GenData.KEEPPROB_ENCODER_QAPDATA
KEEPPROB_DECODER=GM_GenData.KEEPPROB_DECODER_QAPDATA
KEEPPROB_CONV=GM_GenData.KEEPPROB_CONV_QAPDATA
LATENT_DIM=GM_GenData.LATENT_DIM_QAPDATA
REGULAR_RATE=GM_GenData.REGULAR_RATE_QAPDATA

SAVE_ROOT= "trained_models_R1_QAP/QAP_EKpb-"+str(KEEPPROB_ENCODER)+"_DKpb-"+str(KEEPPROB_DECODER)+"_CKpb-"+str(KEEPPROB_CONV)\
           +"_LD-"+str(LATENT_DIM)+"_LR-"+str(LEARNING_RATE)+"_RR-"+str(REGULAR_RATE) + "/" + CATEGORY

if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT)

def feed_dict_generation(queue, idx):
    print(' ############### feed proc start ########################')
    seed = 66
    rand = np.random.RandomState(seed=seed)
    batch_size_tr=1

    # Number of nodes per graph sampled uniformly from this range.
    num_inner_min_max = (10, 11)
    num_outlier_min_max = (0, 11)

    while True:
        inputs, targets, _ = gmc.generate_featured_graphs(
            rand, batch_size_tr, num_inner_min_max, num_outlier_min_max,
            dataset= TRAIN_DATASET)

        queue.put((inputs, targets), block = True, timeout = None)


def train_proc():
    print(' *************** train proc start ********************')
    tf.reset_default_graph()

    seed = 66
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 3
    num_processing_steps_ge = 3

    # Data / training parameters.
    num_training_iterations = 100000
    num_training_epoches = 2000
    theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
    batch_size_tr = 1
    batch_size_ge = 1
    # Number of nodes per graph sampled uniformly from this range.
    num_inner_min_max = (10, 11)
    num_outlier_min_max = (0, 11)

    # Keep probability of encoder and decoder
    keep_prob_encoder = KEEPPROB_ENCODER
    keep_prob_decoder = KEEPPROB_DECODER
    keep_prob_conv=KEEPPROB_CONV
    # Data.
    gmc.NODE_OUTPUT_SIZE = 1

    keep_prob_encoder_ph, keep_prob_decoder_ph = tf.placeholder(shape=None, dtype=tf.float64), tf.placeholder(
        shape=None, dtype=tf.float64)
    keep_prob_conv_ph = tf.placeholder(shape=None, dtype=tf.float64)

    # Input and target placeholders.
    input_ph, target_ph, loss_cof_ph = gmc.create_placeholders(
        rand, batch_size_tr, num_inner_min_max, num_outlier_min_max ,dataset=TRAIN_DATASET)

    # Connect the data to the model.
    # Instantiate the model.
    model = models.EncodeProcessDecode(visfeaType = None,
                                       dynamic_core_num= 0,
                                       edge_output_size=1,
                                       node_output_size=gmc.NODE_OUTPUT_SIZE,
                                       group_output_size = 1)

    # A list of outputs, one per processing step.
    output_ops_tr = model(input_ph, num_processing_steps_tr,keep_prob_encoder_ph, keep_prob_decoder_ph,keep_prob_conv_ph)
    output_ops_ge = model(input_ph, num_processing_steps_ge,keep_prob_encoder_ph, keep_prob_decoder_ph,keep_prob_conv_ph)

    # regularizer loss
    weights_list=[]
    for tv in tf.trainable_variables():
        if tv.name.split("/")[-1][0]=="w":
            weights_list.append(tv)
    regularizer_l1 = tf.contrib.layers.l1_regularizer(REGULAR_RATE, scope=None)
    regular_loss=tf.contrib.layers.apply_regularization(regularizer_l1, weights_list=weights_list)
    regular_loss = tf.cast(regular_loss,tf.float64)
    # Training loss.
    loss_op_tr = gmc.create_loss_ops(target_ph, output_ops_tr, loss_cof_ph)
    loss_op_tr_prev=loss_op_tr
    loss_op_tr=loss_op_tr
    # Test/generalization loss.
    loss_op_ge = gmc.create_loss_ops(target_ph, output_ops_ge, loss_cof_ph)

    # Optimizer.
    lr = LEARNING_RATE
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.maximum(0.0001, tf.train.exponential_decay(lr, global_step, 5, 0.99, staircase = False))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Retrieve all trainable variables you defined in your graph
    tvs = tf.trainable_variables()

    # Creation of a list of variables with the same shape as the trainable ones
    # initialized with zeros
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs if not tv is None]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    # Calls the compute_gradients function of the optimizer to obtain the list of gradients
    gvs = optimizer.compute_gradients(loss_op_tr, tvs)

    # Adds to each element from the list you initialized earlier with zeros its gradient
    # (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs) if not gv[0] is None]

    # Define the training step (part with variable value update)
    train_step = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)], global_step=global_step)

    # set logger
    logger = logging.getLogger("record_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(SAVE_ROOT, "record.log"), mode="w")
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # step_op = optimizer.minimize(loss_op_tr, global_step)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = gmc.make_all_runnable_in_session(input_ph, target_ph)

    #======================================================================================
    # @title Reset session  { form-width: "30%" }

    # This cell resets the Tensorflow session, but keeps the same computational graph.

    try:
        sess.close()
    except NameError:
        pass

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GM_GenData.gpu_memory_fraction)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    last_iteration = 0
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    correct_all_tr_list=[]
    correct_gt_tr_list=[]

    print("# (iteration number), FT (elapsed feed_dict seconds), TT (elapsed training second),"
           "Ltr (training loss), Lge (test/generalization loss), "
           "C_All (test nodes (for all) labeled correctly), "
           "C_GT (test nodes (for groundtruth) labeled correctly), "
           "Sge (test/generalization fraction examples solved correctly)")

    # saver for model
    saver_loss = tf.train.Saver(max_to_keep = 1)
    saver_acc = tf.train.Saver(max_to_keep = 1)
    max_accuracy = 0.0
    min_loss = 1e6
    max_acc = 0.0
    best_result = float("inf")
    start_time = time.time()
    feed_dict_time = 0.0
    training_time = 0.0
    eval_time = 0.0
    best_count = 1
    for epoch in range(num_training_epoches):
        random.shuffle(sample_index)
        # 在run每个batch, 需先将前一个batch所得的累积梯度清零
        sess.run(zero_ops)
        cur_count = 0
        for iteration in range(len(data_list)):
            last_time = time.time() 

            inputs, targets, _ = gmc.generate_featured_graphs(
                rand, batch_size_tr, num_inner_min_max, num_outlier_min_max, use_train_set = False,
                dataset=TRAIN_DATASET, sample_id= sample_index[iteration])

            input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
            target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

            if gmc.NODE_OUTPUT_SIZE == 1:
                loss_cof = target_graphs.nodes * 5.0 + 1.0
            else:
                loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
                loss_cof[:][1] = 5.0

            feed_dict_tr = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof,
                            keep_prob_encoder_ph:keep_prob_encoder,keep_prob_decoder_ph:keep_prob_decoder,
                            keep_prob_conv_ph:keep_prob_conv}

            feed_dict_time = feed_dict_time + time.time() - last_time

            last_time = time.time()

            train_values = sess.run({
                "accum_ops": accum_ops,
                #   "step": step_op,
                "target": target_ph,
                "loss": loss_op_tr,
                "loss_prev":loss_op_tr_prev,
                "outputs": output_ops_tr,
                "learning_rate": learning_rate},
                 feed_dict=feed_dict_tr)

            training_time = training_time + time.time() - last_time

            correct_gt_tr, correct_all_tr, solved_tr, matches_tr, X, _ = gmc.compute_accuracy(
                train_values["target"], train_values["outputs"][-1], use_edges=False)
            if correct_gt_tr < max_accuracy - 0.5:
                print(" **************** exception *****************************************")

            losses_tr.append(train_values["loss"])
            correct_all_tr_list.append(correct_all_tr)
            correct_gt_tr_list.append(correct_gt_tr)

            # test the dataset which only includes one sample
            test_sample_info = torch.load(os.path.join(qaplib.ROOT_PROCESSED_DATA, data_list[sample_index[iteration]] + "_dict.pth"))
            aff_mat = test_sample_info["aff_mat"]
            num_nodes = test_sample_info["A0"].shape[0]
            perm_mat = X[-1].reshape((num_nodes, num_nodes))
            perm_mat = gmc.hungarian(perm_mat)

            perm_mat_left = perm_mat.reshape((1, -1))
            perm_mat_right = perm_mat.reshape((-1, 1))
            res = np.matmul(np.matmul(perm_mat_left, aff_mat), perm_mat_right).astype(np.int).item()

            if res < baseline_res_dict[data_list[sample_index[iteration]]]:
                     cur_count += 1
            logger.info(data_list[sample_index[iteration]] + "  " +str(res))

        logger.info("# {:05d}, FT {:.1f}, TT {:.1f},  Ltr {:.4f}, CAtr {:.4f}, LR {:.5f}".format(
                    epoch, feed_dict_time, training_time, np.mean(np.array(losses_tr)),
                    np.mean(np.array(correct_all_tr_list)), train_values["learning_rate"]))

        losses_tr.clear()
        correct_all_tr_list.clear()

        if cur_count >= best_count:
             best_count = cur_count
             logger.info("**********best_count = " + str(best_count) + "*************")

        sess.run(train_step)
        # evaluation

        # if epoch % 100 == 0:
        #     cur_count = 0
        #     for sid in range(len(data_list)):
        #         feed_dict_ge, raw_graphs = gmc.create_feed_dict(
        #             rand, batch_size_ge, num_inner_min_max, num_outlier_min_max,
        #             "", input_ph, target_ph, loss_cof_ph,
        #             use_train_set=False, dataset=TRAIN_DATASET, sample_id= sid)
        #
        #         feed_dict_ge[keep_prob_encoder_ph] = 1.0
        #         feed_dict_ge[keep_prob_decoder_ph] = 1.0
        #         feed_dict_ge[keep_prob_conv_ph] = 1.0
        #         test_values = sess.run({
        #             "target": target_ph,
        #             "loss": loss_op_ge,
        #             "outputs": output_ops_ge},
        #             feed_dict=feed_dict_ge)
        #
        #         test_sample_info = torch.load(os.path.join(qaplib.ROOT_PROCESSED_DATA, data_list[sid] + "_dict.pth"))
        #
        #         aff_mat = test_sample_info["aff_mat"]
        #         num_nodes = test_sample_info["A0"].shape[0]
        #         perm_mat = test_values["outputs"][-1].nodes.reshape((num_nodes, num_nodes))
        #         perm_mat = gmc.hungarian(perm_mat)
        #
        #         perm_mat_left = perm_mat.reshape((1,-1))
        #         perm_mat_right = perm_mat.reshape((-1,1))
        #         res = np.matmul(np.matmul(perm_mat_left, aff_mat), perm_mat_right).astype(np.int).item()
        #         if res < baseline_res_dict[data_list[sid]]:
        #             cur_count += 1
        #         logger.info(data_list[sid] + "  " +str(res))
        #
        #     if cur_count >= best_count:
        #         best_count = cur_count
        #         logger.info("**********best_count = " + str(best_count) + "*************")

if __name__ == '__main__':

    train_proc()


    print('============ train finished ! ============================')


