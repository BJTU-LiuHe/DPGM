#@title ##### License
# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


#@title Imports  { form-width: "30%" }

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import collections
import itertools
import time

#from graph_nets import graphs
#from graph_nets import utils_np
from graph_nets import utils_featured_graph
from graph_nets import utils_tf
#from graph_nets.demos import models
#import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np
import tensorflow as tf

import Points2D
import CMUHouse
import Willow
import PascalVoc as VOC

###################
# Find the shortest path in a graph\n",
#    "This notebook and the accompanying code demonstrates how to use the Graph Nets library to learn to predict the shortest path between two nodes in graph.\n",
#
#    "The network is trained to label the nodes and edges of the shortest path, given the start and end nodes.\n",
#
#    "After training, the network's prediction ability is illustrated by comparing its output to the true shortest path.
#    Then the network's ability to generalise is tested, by using it to predict the shortest path in similar but larger graphs."
################


#@title ### Install the Graph Nets library on this Colaboratory runtime  { form-width: "60%", run: "auto"}
#@markdown <br>1. Connect to a local or hosted Colaboratory runtime by clicking the **Connect** button at the top-right.<br>2.
# Choose "Yes" below to install the Graph Nets library on the runtime machine with:<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# ```pip install graph_nets```<br> Note, this works both with local and hosted Colaboratory runtimes.

install_graph_nets_library = "No"  #@param ["Yes", "No"]

NODE_OUTPUT_SIZE = 1

if install_graph_nets_library.lower() == "yes":
  print("Installing Graph Nets library with:")
  print("  $ pip install graph_nets\n")
  print("Output message from command:\n")
#  !pip install graph_nets
else:
  print("Skipping installation of Graph Nets library")

#  "If you are running this notebook locally (i.e., not through Colaboratory), you will also need to install a few more dependencies.
#  Run the following on the command line to install the graph networks library, as well as a few other dependencies:\n",
#
#  "pip install graph_nets matplotlib scipy\n",
# =================================

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#@title Helper functions  { form-width: "30%" }

# pylint: disable=redefined-outer-name

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.
NODE_ATTRIBUTE_NAME = "node_attr"

MAX_NODES_PER_GRAPH = 100           # maximum of number of nodes in each graph

def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


def set_diff(seq0, seq1):
  """Return the set difference between 2 sequences as a list."""
  return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
  one_hot = np.eye(max_value)[indices]
  if axis not in (-1, one_hot.ndim):
    one_hot = np.moveaxis(one_hot, -1, axis)
  return one_hot


def get_node_dict(graph, attr):
  """Return a `dict` of node:attribute pairs from a graph."""
  return {k: v[attr] for k, v in graph.node.items()}


# def generate_featured_graph_by_affinity(graph):
#     num_nodes = graph["node_features"].shape[0]
#
#     gidx1 = graph["gidx1"]
#     gidx2 = graph["gidx2"]
#     node_features = np.hstack((graph["node_features"], gidx1, gidx2))
#
#     attr_dicts = dict(gidx1 = gidx1, gidx2 = gidx2)
#
#     node_solution = []
#     for i in range(num_nodes):
#        node_solution.append(graph["solutions"][i])
#
#     return {
#         "node_features": node_features,
#         "node_solution": node_solution,
#         "senders": graph["senders"],
#         "receivers": graph["receivers"],
#         "edge_features": graph["edge_features"],
#         "attr_dicts": attr_dicts
#     }

def featured_graph_to_input_target(graph):
  """Returns 2 graphs with input and target feature vectors for training.

  Args:
    graph: An `nx.DiGraph` instance.

  Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

  Raises:
    ValueError: unknown node type
  """

  def create_feature(attr, fields):
     return np.hstack([np.array(attr[field], dtype=float) for field in fields])

#  input_node_fields = ("affinity", "gidx1", "gidx2")
#  input_edge_fields = ("affinity",)
#  target_node_fields = ("solution",)
#  target_edge_fields = ("solution",)

  gidx1 = np.int32(graph["gidx1"])
  gidx2 = np.int32(graph["gidx2"])
  gidx1_t = np.reshape(gidx1, (len(gidx1), 1), order = 'C')
  gidx2_t = np.reshape(gidx2, (len(gidx2), 1), order = 'C')

  attr_dicts = dict(gidx1 = gidx1, gidx2 = gidx2)
  input_attr_dicts =attr_dicts
  target_attr_dicts = attr_dicts

#  node_features = np.hstack((graph["node_features"], gidx1_t, gidx2_t))
  node_features = graph["node_features"]
  node_solution = graph["solutions"]

  input_attr_dicts["features"] = np.array([0.0])
  target_attr_dicts["features"] = np.array([0.0])

  # generate groups info for target_graph
  n_group1 = np.int(np.max(gidx1) + 1)
  n_group2 = np.int(np.max(gidx2) + 1)

  groups1 = np.zeros((n_group1,1), dtype=np.float)
  groups2 = np.zeros((n_group2,1), dtype=np.float)
  for i in range(len(node_solution)):
      if node_solution[i] :
        groups1[gidx1[i]] = groups1[gidx1[i]] + node_solution[i]
        groups2[gidx2[i]] = groups2[gidx2[i]] + node_solution[i]

  input_attr_dicts["gidx1"] = gidx1
  input_attr_dicts["gidx2"] = gidx2
  input_attr_dicts["groups1"] = np.zeros((n_group1,1), dtype=np.float)
  input_attr_dicts["groups2"] = np.zeros((n_group2,1), dtype=np.float)
  input_attr_dicts["patches1"] = graph["patches1"]
  input_attr_dicts["patches2"] = graph["patches2"]

  target_attr_dicts["gidx1"] = gidx1
  target_attr_dicts["gidx2"] = gidx2
  target_attr_dicts["groups1"] = groups1
  target_attr_dicts["groups2"] = groups2

  if NODE_OUTPUT_SIZE == 1:
      target_node_features = np.zeros((len(node_solution), 1), dtype = np.float)
      for i in range(len(node_solution)) :
         target_node_features[i][0] = node_solution[i]
  else:
      target_node_features = []
      for i in range(len(node_solution)) :
          feature = to_one_hot(node_solution[i].astype(int), NODE_OUTPUT_SIZE)
          target_node_features.append(feature)
      target_node_features = np.array(target_node_features)

  edge_features = {"senders": graph["senders"],
                   "receivers": graph["receivers"],
                   "features": graph["edge_features"]}
  input_graph = utils_featured_graph.FeaturedGraph(node_features, edge_features, input_attr_dicts)
  target_graph = utils_featured_graph.FeaturedGraph(target_node_features, edge_features, target_attr_dicts)

  return input_graph, target_graph


def generate_featured_graphs_by_affinities(affinities) :
    input_graphs = []
    target_graphs = []
    graphs = []
    for i in range(len(affinities)):
        graph = generate_featured_graph_by_affinity(affinities[i])

        input, target = featured_graph_to_input_target(graph)

        input_graphs.append(input)
        target_graphs.append(target)
        graphs.append(graph)

    return input_graphs, target_graphs, graphs


def generate_featured_graphs(rand,
                             batch_size,
                             num_inner_min_max,
                             num_outlier_min_max,
                             visfeaType,
                             use_train_set = True,
                             dataset="") :

    # graphs = Points2D.gen_random_graphs_Points2D(rand,
    #                                                      batch_size,
    #                                                      num_inner_min_max,
    #                                                      num_outlier_min_max)
    # graphs = CMUHouse.gen_random_graphs_CMU(rand,
    #                                                      batch_size,
    #                                                      num_inner_min_max,
    #                                                      num_outlier_min_max)

    if dataset=="Willow":
        graphs, _, _, _ = Willow.gen_random_graphs_Willow(rand,
                                                   batch_size,
                                                   num_inner_min_max,
                                                   num_outlier_min_max,
                                                   feaType = visfeaType,
                                                   use_train_set=use_train_set)
    elif dataset=="Pascal":
        graphs, _, _, _ = VOC.gen_random_graphs_VOC(rand,
                                              batch_size,
                                              num_inner_min_max,
                                              num_outlier_min_max,
                                              feaType = visfeaType)

    input_graphs = []
    target_graphs = []
#    graphs = []
    for i in range(batch_size) :
    #    graph = generate_featured_graph_by_affinity(affinities[i])

        input, target = featured_graph_to_input_target(graphs[i])

        input_graphs.append(input)
        target_graphs.append(target)
    #    graphs.append(graph)


    return input_graphs, target_graphs, graphs



def create_placeholders(rand,
                        batch_size,
                        num_inner_min_max,
                        num_outlier_min_max,
                        visfeaType,dataset):
  """Creates placeholders for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.

  Returns:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.
  """
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs, _ = generate_featured_graphs(
      rand, batch_size, num_inner_min_max, num_outlier_min_max, visfeaType = visfeaType,dataset=dataset)
  input_ph = utils_tf.placeholders_from_weighted_graphs(input_graphs)
  target_ph = utils_tf.placeholders_from_weighted_graphs(target_graphs)
  loss_cof_ph = tf.placeholder(dtype = target_ph.nodes.dtype, shape = target_ph.nodes.shape)
  return input_ph, target_ph, loss_cof_ph


def create_feed_dict_by_graphs(graphs,
                                   input_ph,
                                   target_ph,
                                   loss_cof_ph):

    inputs = []
    targets = []
    for i in range(len(graphs)):
     #   graph = featured_graph_to_input_target(graphs[i])
        input, target = featured_graph_to_input_target(graphs[i])
        inputs.append(input)
        targets.append(target)

    input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
    target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

    if NODE_OUTPUT_SIZE == 1:
        loss_cof = target_graphs.nodes * 5.0 + 1.0
    else:
        loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
        loss_cof[:][1] = 5.0

    feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof}

    return feed_dict, graphs


def create_feed_dict_by_affinities(affinities,
                                   input_ph,
                                   target_ph,
                                   loss_cof_ph):

    inputs, targets, raw_graphs = generate_weighted_graphs_by_affinities(affinities)
    input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
    target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

    if NODE_OUTPUT_SIZE == 1:
        loss_cof = target_graphs.nodes * 5.0 + 1.0
    else:
        loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
        loss_cof[:][1] = 5.0

    feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof}

    return feed_dict, raw_graphs


def create_feed_dict(rand,
                     batch_size,
                     num_inner_min_max,
                     num_outlier_min_max,
                     visfeaType,
                     input_ph,
                     target_ph,
                     loss_cof_ph,
                     use_train_set,
                     dataset):
  """Creates feed_dict for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

  Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
  """
  inputs, targets, raw_graphs = generate_featured_graphs(
      rand, batch_size, num_inner_min_max, num_outlier_min_max,
      visfeaType = visfeaType, use_train_set = use_train_set,dataset=dataset)
  input_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(inputs)
  target_graphs = utils_featured_graph.featured_graphs_to_graphs_tuple(targets)

  if NODE_OUTPUT_SIZE == 1:
    loss_cof = target_graphs.nodes * 5.0  + 1.0
  else:
    loss_cof = np.ones(shape=target_graphs.nodes.shape, dtype=target_graphs.nodes.dtype)
    loss_cof[:][1] = 5.0

  feed_dict = {input_ph: input_graphs, target_ph: target_graphs, loss_cof_ph: loss_cof}

  return feed_dict, raw_graphs



def greedy_mapping(nodes, group_indices):
    x = np.zeros(shape = nodes.shape, dtype = np.int)
    count = 0

    while True:
        idx = np.argmax(nodes)
        if nodes[idx] <= 0.0 :
            break

        nodes[idx] = 0.0
        x[idx] = 1
        count = count + 1

        gidx = group_indices[idx]

        for i in range(len(nodes)):
            if group_indices[i] == gidx :
                nodes[i] = 0.0

    return x, count

def compute_accuracy(target, output, use_nodes=True, use_edges=False):
  """Calculate model accuracy.

  Returns the number of correctly predicted shortest path nodes and the number
  of completely solved graphs (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.
    use_nodes: A `bool` indicator of whether to compute node accuracy or not.
    use_edges: A `bool` indicator of whether to compute edge accuracy or not.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.

  Raises:
    ValueError: Nodes or edges (or both) must be used
  """
  if not use_nodes and not use_edges:
    raise ValueError("Nodes or edges (or both) must be used")

  tdds = utils_featured_graph.graphs_tuple_to_data_dicts(target)
  odds = utils_featured_graph.graphs_tuple_to_data_dicts(output)

  cs_all = []
  ss_all = []
  cs_gt = []
  num_matches = 0
  Xs = []
  gXs = []

  if NODE_OUTPUT_SIZE == 1:
      for td, od in zip(tdds, odds):
        xn = td["nodes"].astype(np.int)
        yn, num = greedy_mapping(od["nodes"], od["group_indices_1"])
        num_matches = num_matches + num

        Xs.append(yn)
        gXs.append(xn)

        c_all = (xn == yn)
        s_all = np.all(c_all)
        cs_all.append(c_all)
        ss_all.append(s_all)

        c_gt = 0
        for i in range(len(xn)):
            if xn[i] == 1 and xn[i] == yn[i] :
                c_gt = c_gt + 1
        if np.sum(xn) > 0:
            c_gt = c_gt / np.sum(xn)
        else:
            c_gt = 0
        cs_gt.append(c_gt)
  else:
      for td, od in zip(tdds, odds):
        tlabels = np.argmax(td["nodes"], axis = 1)
        olabels = np.argmax(od["nodes"], axis = 1)
        olabels, _ = greedy_mapping(olabels, od["group_indices_1"])
        num_matches = num_matches + np.sum(olabels)

        Xs.append(olabels)
        gXs.append(tlabels)

        c_all = (tlabels == olabels)
        s_all = np.all(c_all)
        cs_all.append(c_all)
        ss_all.append(s_all)

        c_gt = tlabels.dot(c_all)
        c_gt = np.sum(c_gt) / np.sum(tlabels)
        cs_gt.append(c_gt)

  correct_gt = np.mean(np.array(cs_gt))
  correct_all = np.mean(np.concatenate(cs_all, axis=0))
  solved = np.mean(np.stack(ss_all))
  return correct_gt, correct_all, solved, num_matches, Xs, gXs


def create_loss_ops(target_op, output_ops, loss_cof):
    output_op = output_ops[-1]
    target_nodes=target_op.nodes
    output_nodes=output_op.nodes
    group_cof = 0.1
    label_cof = loss_cof
    if NODE_OUTPUT_SIZE == 1:
        loss_nodes = tf.losses.mean_squared_error(label_cof * target_op.nodes, label_cof * output_op.nodes)
        loss_groups_1 = tf.losses.mean_squared_error(target_op.groups_1, output_op.groups_1)
        loss_ops = loss_nodes + group_cof * loss_groups_1
    else:
        loss_nodes = tf.losses.softmax_cross_entropy(label_cof * target_op.nodes, label_cof * output_op.nodes)
        #groups_1 = output_op.groups_1[:][1]
        #loss_groups_1 = tf.losses.mean_squared_error(target_op.groups_1, groups_1)
        loss_ops = loss_nodes

#  loss_ops = (1.0 - loss_cof_ph) * loss_nodes + loss_cof_ph * loss_groups_1

#    loss_ops = loss_nodes + 0.01 * loss_groups_1

    return loss_ops


def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


