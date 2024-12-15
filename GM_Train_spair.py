from graph_nets.demos import models
from graph_nets import utils_featured_graph

import time
#import threading
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import GM_Core_featured as gmc
import GM_GenData
import torch
import logging

import os
from Spair71k import test_dataset_dict

os.environ['CUDA_VISIBLE_DEVICES'] = GM_GenData.GPU_ID

assert GM_GenData.DATASET == "Spair71k"

NODE_VISFEA_TYPE = GM_GenData.VISFEA_TYPE_BBGM
TRAIN_DATASET = GM_GenData.DATASET
LEARNING_RATE=GM_GenData.LEARNING_RATE

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
num_categories=18

KEEPPROB_ENCODER=GM_GenData.KEEPPROB_ENCODER_SPAIR71K
KEEPPROB_DECODER=GM_GenData.KEEPPROB_DECODER_SPAIR71K
KEEPPROB_CONV=GM_GenData.KEEPPROB_CONV_SPAIR71K
MEAN_POOLING_INT=GM_GenData.MEAN_POOLING_INTERVAL_SPAIR71K
LATENT_DIM=GM_GenData.LATENT_DIM_SPAIR71K
REGULAR_RATE=GM_GenData.REGULAR_RATE_SPAIR71K
if GM_GenData.GRAPH_MODE_SPAIR71K=="DEL":
    GRAPH_MODE="DEL"
elif GM_GenData.GRAPH_MODE_SPAIR71K=="KNN":
    GRAPH_MODE=str(GM_GenData.NUM_K_SPAIR71K)+"NN"


SAVE_ROOT= "trained_models/"+GM_GenData.DATASET


def feed_dict_generation(queue, idx):
    print(' ############### feed proc start ########################')
    seed = 66
    rand = np.random.RandomState(seed=seed)
    batch_size_tr=20

    # Number of nodes per graph sampled uniformly from this range.
    num_inner_min_max = (10, 11)
    num_outlier_min_max = (0, 11)

    while True:
        inputs, targets, _ = gmc.generate_featured_graphs(
            rand, batch_size_tr, num_inner_min_max, num_outlier_min_max,
            visfeaType = NODE_VISFEA_TYPE,dataset=TRAIN_DATASET)

        queue.put((inputs, targets), block = True, timeout = None)

        #print('#{}: ++++++++++++ queue.put +++++++++++++++++++'.format(idx))

def train_proc(queue):
    result_dict = torch.load("result_dict.pth")
    print(' *************** train proc start ********************')
    tf.reset_default_graph()

    seed = 66
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 3
    num_processing_steps_ge = 3

    # Data / training parameters.
    num_training_iterations = 500000*2
    theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
    batch_size_tr = 20
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
        rand, batch_size_tr, num_inner_min_max, num_outlier_min_max, visfeaType = NODE_VISFEA_TYPE,dataset=TRAIN_DATASET)

    # Connect the data to the model.
    # Instantiate the model.
    model = models.EncodeProcessDecode(visfeaType = NODE_VISFEA_TYPE,
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
    regularizer_l1 = tf.contrib.layers.l2_regularizer(REGULAR_RATE, scope=None)
    regular_loss=tf.contrib.layers.apply_regularization(regularizer_l1, weights_list=weights_list)
    regular_loss = tf.cast(regular_loss,tf.float64)
    # Training loss.
    loss_op_tr = gmc.create_loss_ops(target_ph, output_ops_tr, loss_cof_ph)
    loss_op_tr_prev=loss_op_tr
    loss_op_tr=loss_op_tr+regular_loss
    # Test/generalization loss.
    loss_op_ge = gmc.create_loss_ops(target_ph, output_ops_ge, loss_cof_ph)

    # Optimizer.
    lr = LEARNING_RATE
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.maximum(0.0001, tf.train.exponential_decay(lr, global_step, 1000, 0.995, staircase = False))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    # set logger
    logger = logging.getLogger("record_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(SAVE_ROOT, "record.log"),mode= "w")
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


    step_op = optimizer.minimize(loss_op_tr, global_step)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = gmc.make_all_runnable_in_session(input_ph, target_ph)

    #======================================================================================
    # @title Reset session  { form-width: "30%" }

    # This cell resets the Tensorflow session, but keeps the same computational graph.

    try:
        sess.close()
    except NameError:
        pass

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GM_GenData.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())


    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    correct_all_tr_list=[]
    correct_gt_tr_list=[]
    losses_ge = []
    corrects_ge = []
    solveds_ge = []
    correct_gt_ge_list=[]
    matches_ge_list=[]

    print("# (iteration number), FT (elapsed feed_dict seconds), TT (elapsed training second),"
           "Ltr (training loss), Lge (test/generalization loss), "
        #   "Ctr (training fraction nodes/edges labeled correctly), "
        #   "Str (training fraction examples solved correctly), "
           "C_All (test nodes (for all) labeled correctly), "
           "C_GT (test nodes (for groundtruth) labeled correctly), "
           "Sge (test/generalization fraction examples solved correctly)")

    # saver for model
    saver_loss = tf.train.Saver(max_to_keep = 1)
    saver_acc = tf.train.Saver(max_to_keep = 1)
    max_accuracy = 0.0
    min_loss = 1e6
    max_acc = 0.0

    start_time = time.time()
    last_log_time = start_time
    feed_dict_time = 0.0
    training_time = 0.0
    eval_time = 0.0
    save_best_dict = dict()
    save_bad_dict = dict()

    for iteration in range(last_iteration, num_training_iterations):
        last_time = time.time()

        inputs, targets = queue.get(block = True, timeout = None)

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
            "step": step_op,
            "target": target_ph,
            "loss": loss_op_tr,
            "loss_prev":loss_op_tr_prev,
            "outputs": output_ops_tr,
            "learning_rate": learning_rate},
             feed_dict=feed_dict_tr)

        training_time = training_time + time.time() - last_time

        correct_gt_tr, correct_all_tr, solved_tr, matches_tr, _, _ = gmc.compute_accuracy(
            train_values["target"], train_values["outputs"][-1], use_edges=False)
        if correct_gt_tr < max_accuracy - 0.5:
            print(" **************** exception *****************************************")
        losses_tr.append(train_values["loss"])
        correct_all_tr_list.append(correct_all_tr)
        correct_gt_tr_list.append(correct_gt_tr)

        the_time = time.time()
        if iteration % 100 ==0 :
            logger.info("# {:05d}, FT {:.1f}, TT {:.1f},  Ltr {:.4f}, CAtr {:.4f}, CGtr {:.4f}, LR {:.5f}".format(
                iteration, feed_dict_time, training_time, np.mean(np.array(losses_tr)),
                np.mean(np.array(correct_all_tr_list)),
                np.mean(np.array(correct_gt_tr_list)),
                train_values["learning_rate"]))


            losses_tr.clear()
            correct_all_tr_list.clear()
            correct_gt_tr_list.clear()
        if iteration % 1000 == 0:
            last_time = the_time
            accuracy_dict = {}
            for category in categories:
                accuracy_dict[category] = []

            for category_id in range(num_categories):
                best_list, bad_list = [], []
                for idx in range(len(test_dataset_dict[categories[category_id]])):
                    feed_dict_ge, raw_graphs = gmc.create_feed_dict(
                        rand, batch_size_ge, num_inner_min_max, num_outlier_min_max,
                        NODE_VISFEA_TYPE, input_ph, target_ph, loss_cof_ph,
                        use_train_set=False, dataset=TRAIN_DATASET, category= category_id,
                        sample_id= idx)

                    feed_dict_ge[keep_prob_encoder_ph]=1.0
                    feed_dict_ge[keep_prob_decoder_ph] = 1.0
                    feed_dict_ge[keep_prob_conv_ph]=1.0
                    test_values = sess.run({
                        "target": target_ph,
                        "loss": loss_op_ge,
                        "outputs": output_ops_ge},
                        feed_dict=feed_dict_ge)


                    correct_gt_ge, correct_all_ge, solved_ge, matches_ge, _, _ = gmc.compute_accuracy(
                        test_values["target"], test_values["outputs"][-1], use_edges=False)

                    accuracy_dict[categories[category_id]].append(correct_gt_ge)

                    losses_ge.append(test_values["loss"])
                    corrects_ge.append(correct_all_ge)
                    solveds_ge.append(solved_ge)
                    correct_gt_ge_list.append(correct_gt_ge)
                    matches_ge_list.append(matches_ge)
                    logged_iterations.append(iteration)


            accuracy_avg = []
            for key, value in accuracy_dict.items():
                accuracy_avg.append(np.mean(value))
            accuracy_avg = np.mean(accuracy_avg).item()

            elapsed = time.time() - start_time
            if np.mean(np.array(losses_ge)) < min_loss:
                save_file=SAVE_ROOT+"/best_model_loss"
                saver_loss.save(sess, save_file, global_step=iteration)
                min_loss = np.mean(np.array(losses_ge))
            if accuracy_avg > max_acc:
                save_file=SAVE_ROOT+"/best_model_acc"
                saver_acc.save(sess, save_file, global_step=iteration)
                max_acc = accuracy_avg
                torch.save(accuracy_dict, os.path.join(SAVE_ROOT, "accuracy_dict.pth"))

                for key, value in accuracy_dict.items():
                    logger.info(key + ":" + str(np.mean(value).item()))

                logger.info("avg:" + str(accuracy_avg))

            eval_time = eval_time + time.time() - last_time
            logger.info(
                "# {:05d}, T {:.1f}, FT {:.1f}, TT {:.1f}, ET {:.1f}, Ltr {:.4f}, CAtr {:.4f}, CGtr {:.4f} Lge {:.4f},  CAge {:.4f}, CGge {:.4f}, NEG {:f}, LR {:.5f}".format(
                    iteration, elapsed, feed_dict_time, training_time, eval_time, np.mean(np.array(losses_tr)),
                    np.mean(np.array(correct_all_tr_list)),
                    np.mean(np.array(correct_gt_tr_list)), np.mean(np.array(losses_ge)), np.mean(np.array(corrects_ge)),
                    np.mean(np.array(correct_gt_ge_list)), np.mean(np.array(matches_ge_list)),
                    train_values["learning_rate"]))

            losses_tr.clear()
            losses_ge.clear()
            corrects_ge.clear()
            solveds_ge.clear()
            correct_all_tr_list.clear()
            correct_gt_tr_list.clear()
            correct_gt_ge_list.clear()
            matches_ge_list.clear()
            logged_iterations.clear()

if __name__ == '__main__':

    queue = mp.Queue(5)
    #
    #
    num_feed_dict_thread = 5#5
    feed_dict_procs = []
    for i in range(num_feed_dict_thread):
        p = mp.Process(target = feed_dict_generation, args = (queue, i))
        p.start()
        feed_dict_procs.append(p)

    print(' ********** feed process started ********************')

    record = []
    tp = mp.Process(target = train_proc, args = (queue,))
    print('tp = ')
    tp.start()
    tp.join()
    record.append(tp)

    print('============ train finished ! ============================')


