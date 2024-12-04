#import torch
import numpy as np

class GraphData():
    def __init__(self, feat_path, knn_graph_path, label_path, k_at_hop=[200,10]):
        self.features = np.load(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:, :k_at_hop[0] + 1]
        self.labels = np.load(label_path)
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop

    def __len__(self):
        return self.num_samples


    def __getitem__(self):
        hops_one = []  # hop_one的节点200个
        hops_set = []  #hop_one+hop_two节点
        for nodeindex in self.knn_graph:
            hops_one.append([])
            for j in self.knn_graph[nodeindex[0]][1:self.k_at_hop[0]+1]:
                hops_one[nodeindex[0]].append(j)

        for nodeindex in self.knn_graph:
            hops_set.append([])
            for one_hop in hops_one[nodeindex[0]]:
                for j in self.knn_graph[one_hop][1:self.k_at_hop[1]+1]:
                    hops_set[nodeindex[0]].append(j)

        for nodeindex in self.knn_graph:
            hops_set[nodeindex[0]]=list(set(hops_set[nodeindex[0]]+hops_one[nodeindex[0]]))

        edge_index=[]
        edge_index.append([])
        edge_index.append([])
        edge_attr=[]
        edge_labels=[]
        #A=torch.zeros(len(self.features), len(self.features))
        A = []
        for nodeindex in self.knn_graph:
            neighbors = self.knn_graph[nodeindex[0], 1:11]
            for n in neighbors:
                if n in hops_set[nodeindex[0]]:
                    edge_index[0].append(nodeindex[0])  # target_node
                    edge_index[1].append(n)  # src_node
                    edge_attr.append(self.features[n]-self.features[nodeindex[0]])
                    #edge_index[1].append(nodeindex[0])  # src_node
                    #edge_index[0].append(n)  # target_node
                    #edge_attr.append(self.features[nodeindex[0]]-self.features[n])
                    if (self.labels[n]==self.labels[nodeindex[0]]):
                        edge_labels.append(1)
                        #edge_labels.append(1)
                    else:
                        edge_labels.append(0)
                        #edge_labels.append(0)
                    #A[nodeindex[0],n]=1
                    #A[n,nodeindex[0]] = 1


        edge_index = np.array(edge_index)
        edge_attr = np.array(edge_attr)
        edge_labels = np.array(edge_labels)
        x = np.array(self.features)

    #    edge_index = torch.tensor(edge_index, dtype=torch.long)
    #    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    #    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    #    x= torch.tensor(self.features, dtype=torch.float)
    #    D = A.sum(1, keepdim=True)  # 度矩阵，对角线上元素依次为各个顶点的度，计算每一行元素的和，并且保持二维特性，类似这样array([[3], [7]])
    #    A = A.div(D)  # 拉普拉斯矩阵D-A

        return x,edge_attr,edge_index,edge_labels,A


# if __name__ == '__main__':
#     mydata=GraphData('C:/Users/fangfang/Desktop/研究生学习/科研/我/code/linked-based/facedata/512.fea.npy','C:/Users/fangfang/Desktop/研究生学习/科研/我/code/linked-based/facedata/knn.graph.512.bf.npy','C:/Users/fangfang/Desktop/研究生学习/科研/我/code/linked-based/facedata/512.labels.npy')
#     x, edge_attr, edge_index,edge_labels,A=mydata.__getitem__()
#     row, col = edge_index
#     ee=x[row]
#     print(x.shape)
#     print(ee.shape)
#     print(edge_index.shape)
# '''#聚合逻辑
#     row,col=edge_index
#     for node_i in range(x.size(0)):
#         index_i=0
#         x_i=torch.zeros(512).cuda()
#         num_i=0
#         for edge_i in row:
#             if node_i==edge_i:
#                 x_i=x_i+edge_attr[index_i]
#             index_i = index_i + 1
#             num_i=num_i+1
#         x[node_i]=x[node_i]+x_i/num_i
#
#     for edge_j in range(edge_index.size(1)):
#         e_j = torch.zeros(512)
#         #print(edge_j)
#         a=row[edge_j]
#         b=col[edge_j]
#         e_j=(x[a]+x[b])/2
#         edge_attr=edge_attr+e_j'''
#     #print(x[0])
#     #print(edge_index.shape)#[2*边数]
#     #print(x.shape)#[结点数*512]
#     #print(edge_attr.shape[0])*[边数*512]
