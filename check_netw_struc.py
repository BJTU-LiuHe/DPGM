import tensorflow as tf

graph=tf.Graph()
with graph.as_default() as graph:
    tf.train.import_meta_graph("trained_models/Pascal_EKpb-1.0_DKpb-1.0_LR-0.0003-preVGG16/best_model_acc-40300.meta")

with tf.Session(graph=graph) as sess:
    file_writer=tf.summary.FileWriter(logdir="log/netw_struc",graph=graph)