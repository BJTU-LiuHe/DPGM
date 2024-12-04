from graph_nets.demos import models
from graph_nets import utils_featured_graph

import time
#import threading
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import GM_Core_featured as gmc
import GM_GenData


import os
os.environ['CUDA_VISIBLE_DEVICES'] = GM_GenData.GPU_ID

NODE_VISFEA_TYPE = GM_GenData.VISFEA_TYPE_BBGM
TRAIN_DATASET = GM_GenData.DATASET
LEARNING_RATE=GM_GenData.LEARNING_RATE

if TRAIN_DATASET=="Willow":
    num_categories=5
    num_batch_ge=2
    KEEPPROB_ENCODER=GM_GenData.KEEPPROB_ENCODER_WILLOW
    KEEPPROB_DECODER=GM_GenData.KEEPPROB_DECODER_WILLOW
    KEEPPROB_CONV=GM_GenData.KEEPPROB_CONV_WILLOW
    MEAN_POOLING_INT=GM_GenData.MEAN_POOLING_INTERVAL_WILLOW
    LATENT_DIM=GM_GenData.LATENT_DIM_WILLOW
    REGULAR_RATE=GM_GenData.REGULAR_RATE_WILLOW
    if GM_GenData.GRAPH_MODE_WILLOW=="DEL":
        GRAPH_MODE="DEL"
    elif GM_GenData.GRAPH_MODE_WILLOW=="KNN":
        GRAPH_MODE=str(GM_GenData.NUM_K_WILLOW)+"NN"
elif TRAIN_DATASET=="Spair17k":
    num_categories = 20
    num_batch_ge = 10
    KEEPPROB_ENCODER = GM_GenData.KEEPPROB_ENCODER_SPAIR71K
    KEEPPROB_DECODER = GM_GenData.KEEPPROB_DECODER_SPAIR71K
    KEEPPROB_CONV = GM_GenData.KEEPPROB_CONV_SPAIR71K
    MEAN_POOLING_INT = GM_GenData.MEAN_POOLING_INTERVAL_SPAIR71K
    LATENT_DIM = GM_GenData.LATENT_DIM_SPAIR71K
    REGULAR_RATE = GM_GenData.REGULAR_RATE_SPAIR71K
    if GM_GenData.GRAPH_MODE_SPAIR71K == "DEL":
        GRAPH_MODE = "DEL"
    elif GM_GenData.GRAPH_MODE_SPAIR71K == "KNN":
        GRAPH_MODE = str(GM_GenData.NUM_K_SPAIR71K) + "NN"
else:
    num_categories=20
    num_batch_ge = 10
    KEEPPROB_ENCODER = GM_GenData.KEEPPROB_ENCODER_PASCAL
    KEEPPROB_DECODER = GM_GenData.KEEPPROB_DECODER_PASCAL
    KEEPPROB_CONV=GM_GenData.KEEPPROB_CONV_PASCAL
    MEAN_POOLING_INT = GM_GenData.MEAN_POOLING_INTERVAL_PASCAL
    LATENT_DIM=GM_GenData.LATENT_DIM_PASCAL
    REGULAR_RATE = GM_GenData.REGULAR_RATE_PASCAL
    if GM_GenData.GRAPH_MODE_PASCAL=="DEL":
        GRAPH_MODE="DEL"
    elif GM_GenData.GRAPH_MODE_PASCAL=="KNN":
        GRAPH_MODE=str(GM_GenData.NUM_K_PASCAL)+"NN"


SAVE_ROOT= "trained_models/new_sinkhorn_pos_w1_"+GM_GenData.DATASET+"_EKpb-"+str(KEEPPROB_ENCODER)+"_DKpb-"+str(KEEPPROB_DECODER)+"_CKpb-"+str(KEEPPROB_CONV)\
           +"_MPI-"+str(MEAN_POOLING_INT)+"_LD-"+str(LATENT_DIM)+"_LR-"+str(LEARNING_RATE)+"_GM-"+GRAPH_MODE+"_RR-"+str(REGULAR_RATE)+"-"+NODE_VISFEA_TYPE+"_PB-5"

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
    print(' *************** train proc start ********************')
    tf.reset_default_graph()

    seed = 66
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 10
    num_processing_steps_ge = 10

    # Data / training parameters.
    num_training_iterations = 100000
    theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
    batch_size_tr = 20
    batch_size_ge = 20
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
    regularizer_l1 = tf.contrib.layers.l1_regularizer(REGULAR_RATE, scope=None)
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
    learning_rate = tf.maximum(0.0008, tf.train.exponential_decay(lr, global_step, 1000, 0.99, staircase = False))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # save path
    save_path=SAVE_ROOT+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    record_file=save_path+"records.txt"
    fid_record=open(record_file,"a")

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

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GM_GenData.gpu_memory_fraction)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # config = tf.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session()
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

    # #load trained model
    # model_file = "trained_models/truncation/reload_new_sinkhorn_pos_w1_Pascal_EKpb-0.9_DKpb-0.9_CKpb-1.0_MPI-2_LD-128_LR-0.002_GM-DEL_RR-0.0-BBGM/best_model_acc-22000"
    # saver_loss.restore(sess, model_file)
    # last_iteration = int(float(model_file.split("-")[-1]))


    start_time = time.time()
    last_log_time = start_time
    feed_dict_time = 0.0
    training_time = 0.0
    eval_time = 0.0
    for iteration in range(last_iteration, num_training_iterations):
        last_iteration = iteration
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

        # outputs_tr = train_values["outputs"][0]
        # torch.save(outputs_tr,"./debug/graph_data.pth")
        # output=train_values["outputs"][-1].nodes.reshape((-1,10,10))[0,:,:]
        # target_nodes=train_values["target"].nodes.reshape((-1,10,10))[0,:,:]
        training_time = training_time + time.time() - last_time

        correct_gt_tr, correct_all_tr, solved_tr, matches_tr, _, _ = gmc.compute_accuracy(
            train_values["target"], train_values["outputs"][-1], use_edges=False)
        if correct_gt_tr < max_accuracy - 0.5:
            print(" **************** exception *****************************************")
        # if train_values["loss"]==np.nan:
        #     print("the current loss is nan, iteration = ",iteration)
        #     break
        # print(train_values["outputs"][-1].nodes[:input_graphs.n_node[0]].reshape(int(np.sqrt(input_graphs.n_node[0])),-1))
        # print(input_graphs.nodes)
        # # print(train_values["outputs"])
        # print(train_values["outputs"][-1].nodes.reshape(-1,10,10)[0,:,:])
        # print(np.max(train_values["outputs"][-1].nodes),np.min(train_values["outputs"][-1].nodes),train_values["loss"])
        losses_tr.append(train_values["loss"])
        correct_all_tr_list.append(correct_all_tr)
        correct_gt_tr_list.append(correct_gt_tr)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        #if elapsed_since_last_log > log_every_seconds:
        if iteration % 100 ==0 and (iteration < 10000):
            record_str = "# {:05d}, FT {:.1f}, TT {:.1f},  Ltr {:.4f}, CAtr {:.4f}, CGtr {:.4f}, LR {:.5f}".format(
                iteration, feed_dict_time, training_time, np.mean(np.array(losses_tr)),
                np.mean(np.array(correct_all_tr_list)),
                np.mean(np.array(correct_gt_tr_list)),
                train_values["learning_rate"])
            fid_record.writelines(record_str + "\n")
            print(record_str)
            losses_tr.clear()
            correct_all_tr_list.clear()
            correct_gt_tr_list.clear()
        if iteration % 10000 == 0 and iteration>=10000:
            last_time = the_time
            for category in range(num_categories):
                for _ in range(20):
                    for _ in range(num_batch_ge):
                        feed_dict_ge, raw_graphs = gmc.create_feed_dict(
                            rand, batch_size_ge, num_inner_min_max, num_outlier_min_max,
                            NODE_VISFEA_TYPE, input_ph, target_ph, loss_cof_ph,
                            use_train_set=False,dataset=TRAIN_DATASET,category=category)

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

                        # losses_tr.append(train_values["loss"])
                        losses_ge.append(test_values["loss"])
                        corrects_ge.append(correct_all_ge)
                        solveds_ge.append(solved_ge)
                        correct_gt_ge_list.append(correct_gt_ge)
                        matches_ge_list.append(matches_ge)
                        logged_iterations.append(iteration)

            elapsed = time.time() - start_time
            if np.mean(np.array(losses_ge)) < min_loss:
                save_file=save_path+"best_model_loss"
                saver_loss.save(sess, save_file, global_step=iteration)
                min_loss = np.mean(np.array(losses_ge))
                # max_acc = correct_gt_ge
            if np.mean(np.array(correct_gt_ge_list)) > max_acc:
                save_file=save_path+"best_model_acc"
                saver_acc.save(sess, save_file, global_step=iteration)
                # min_loss = test_values["loss"]
                max_acc = np.mean(np.array(correct_gt_ge_list))

            eval_time = eval_time + time.time() - last_time
            record_str = "# {:05d}, T {:.1f}, FT {:.1f}, TT {:.1f}, ET {:.1f}, Ltr {:.4f}, CAtr {:.4f}, CGtr {:.4f} Lge {:.4f},  CAge {:.4f}, CGge {:.4f}, NEG {:f}, LR {:.5f}".format(
                iteration, elapsed, feed_dict_time, training_time, eval_time, np.mean(np.array(losses_tr)), np.mean(np.array(correct_all_tr_list)),
                np.mean(np.array(correct_gt_tr_list)),np.mean(np.array(losses_ge)), np.mean(np.array(corrects_ge)),
                np.mean(np.array(correct_gt_ge_list)), np.mean(np.array(matches_ge_list)), train_values["learning_rate"])
            # record_str="# {:05d}, T {:.1f}, FT {:.1f}, TT {:.1f}, ET {:.1f}, Ltr {:.4f}, CAtr {:.4f}, CGtr {:.4f} Lge {:.4f},  CAge {:.4f}, CGge {:.4f}, NEG {:d}, LR {:.5f}".format(
            #         iteration, elapsed, feed_dict_time, training_time, eval_time, train_values["loss"], correct_all_tr, correct_gt_tr,
            #         test_values["loss"], correct_all_ge, correct_gt_ge, matches_ge, train_values["learning_rate"])


            fid_record.writelines(record_str+"\n")
            print(record_str)
            losses_tr.clear()
            losses_ge.clear()
            corrects_ge.clear()
            solveds_ge.clear()
            correct_all_tr_list.clear()
            correct_gt_tr_list.clear()
            correct_gt_ge_list.clear()
            matches_ge_list.clear()
            logged_iterations.clear()

    fid_record.close()

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


