import os
import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import itertools
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def get_target_node_label(data,model,target_node):
    model.eval()
    with torch.no_grad():
        pred = model(data).softmax(-1)
        pred = pred[target_node]
        pred = pred.argmax(-1)
        print(pred)
def get_target_line_node_label(model, dataset, target_node):
    model.eval()
    with torch.no_grad():
        pass


def find_subgraph_in_line_graph(indices, edge_index, data):
    """
    Given a node indices on the original graph, return its subgraph node indices in line graph
    indices: array-like, list of node indices
    edge_index: torch.tensor, edge index tensor of line graph
    line_graph_dict: dictionary key=(src, tgt), value= node_idx in line graph
    """
    # Convert edge_index to list of tuples for easy lookup
    edge_tuples = list(map(tuple, edge_index.t().tolist()))

    # Find subgraph in original graph
    subgraph_indices, subgraph_edge_index = subgraph(indices, edge_index, num_nodes=None, relabel_nodes=True)

    # map edge indices in edge_index to line graph
    line_subgraph_indices = [edge_tuples.index(tuple(edge)) for edge in subgraph_edge_index.t().tolist()]

    return line_subgraph_indices, subgraph_indices

# 写一个函数，把上面的步骤封装起来
def get_target_node_score(model, dataset, target_node):
    # 步骤1：在原始图中找到目标节点，生成两跳子图，找到目标节点相邻的边
    subset, edge_index, _, _ = k_hop_subgraph(target_node, 2, dataset.data.edge_index,
                                              relabel_nodes=True)
    # 步骤2：生成目标节点的两跳子图数据
    subgraph_data = Data(x=dataset.data.x[subset], edge_index=edge_index)
    # 步骤3：将两跳子图映射到线图中的子图
    sub = nx.DiGraph()
    sub.add_edges_from(edge_index.t().tolist())
    LG = nx.line_graph(sub)
    # 步骤4：将线图子图转换成Data格式,计算目标节点的预测得分
    line_graph_edges = [[edge[0][0], edge[0][1]] for edge in LG.edges()]
    line_graph_edge_index = torch.tensor(line_graph_edges, dtype=torch.long, device=device).t().contiguous()
    unique_nodes = set()
    for edge in LG.edges():
        unique_nodes.update(edge)
    # 创建节点映射字典，将原始节点映射到新的索引
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    node_to_original_edge = {idx: edge for idx, edge in enumerate(LG.edges())}
    edge_index = torch.tensor([[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in LG.edges()],
                              dtype=torch.long).t().contiguous()
    lg_subgraph_data = Data(edge_index=edge_index,
                            x=torch.ones((len(unique_nodes), 1)),  # 所有元素都为1
                            y=None)  # 在这里设置你的目标标签，如果有的话
    # 步骤5：计算目标节点的预测得分
    edge_to_find = (0, 1)
    target_node_line = node_mapping[edge_to_find]
    with torch.no_grad():  # 不计算梯度，因为这是预测阶段
        predicted_logits = model(lg_subgraph_data.x, lg_subgraph_data.edge_index)
    # 3. 获取目标节点的预测得分
    target_node_score = predicted_logits[target_node_line]
    # 如果模型输出是概率，可以使用 softmax 转换
    target_node_score = F.softmax(target_node_score, dim=0)
    return target_node_score
