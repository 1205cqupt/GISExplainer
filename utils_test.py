import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import itertools
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import os
import json
"""
    来自DIG的预备工作函数
"""

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_dataset(dataset, num_classes):
    indices = []
    train_percent = 0.7
    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset



import re
from collections import OrderedDict

from torch_geometric import __version__

# dig自带的，当前版本没有就复制下来了
def compatible_state_dict(state_dict):
    comp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        comp_key = key
        comp_value = value
        if int(__version__[0]) >= 2:
            comp_key = re.sub(r'conv(1|s.[0-9]).weight', 'conv\g<1>.lin.weight', key)
            if comp_key != key:
                comp_value = value.T
        if comp_key != key:
            comp_state_dict[key] = value
        comp_state_dict[comp_key] = comp_value
    return comp_state_dict

# 检查有没有对应的预训练模型，没有就去下一个
def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)

############################### 线图 ######################################
# 没用了
def split_line_data(data, train_ratio=0.7, test_ratio=0.15):
    num_nodes = data.num_nodes
    num_train = int(num_nodes * train_ratio)
    num_test = int(num_nodes * test_ratio)

    # 创建一个随机排列的节点索引
    perm = torch.randperm(num_nodes)
    train_index = perm[:num_train]
    val_index = perm[num_train:-num_test]
    test_index = perm[-num_test:]

    # 创建训练、验证和测试掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    # 创建一个新的Data对象，包含训练、验证和测试掩码
    splitted_data = Data(x=data.x, edge_index=data.edge_index, num_classes=data.num_classes)
    splitted_data.train_mask = train_mask
    splitted_data.val_mask = val_mask
    splitted_data.test_mask = test_mask

    return splitted_data

# VIC
def create_line_graph_data_with_nx(dataset):
    data = dataset.data
    edge_index = data.edge_index
    num_edges = data.edge_index.size(1)
    device = edge_index.device
    # 导入原始图
    G = nx.DiGraph()
    G.add_edges_from(edge_index.t().cpu().numpy())
    # 创建线图
    LG = nx.line_graph(G)
    # 将线图转换为PyG数据对象
    line_graph_edges = [[edge[0][0], edge[0][1]] for edge in LG.edges()]
    line_graph_edge_index = torch.tensor(line_graph_edges, dtype=torch.long, device=device).t().contiguous()

    next_label = 0
    label_combinations = {}     # 标签组合
    line_graph_labels = []
    for edge in LG.nodes():
        source, target = edge
        label_pair = tuple(sorted((data.y[source].item(), data.y[target].item())))
        if label_pair not in label_combinations:
            label_combinations[label_pair] = next_label
            next_label += 1
        line_graph_labels.append(label_combinations[label_pair])

    line_graph_data = Data(x=torch.eye(num_edges, device=device), edge_index=line_graph_edge_index)
    line_graph_data.y = torch.tensor(line_graph_labels, dtype=torch.long, device=device)
    # 创建新的mask
    line_graph_data.train_mask = data.train_mask[edge_index[0]] & data.train_mask[edge_index[1]]
    line_graph_data.val_mask = data.val_mask[edge_index[0]] & data.val_mask[edge_index[1]]
    line_graph_data.test_mask = data.test_mask[edge_index[0]] & data.test_mask[edge_index[1]]

    return line_graph_data, line_graph_edge_index, label_combinations
def create_line_graph_data_with_nx(data):
    edge_index = data.edge_index
    num_edges = data.edge_index.size(1)
    device = edge_index.device
    # 导入原始图
    G = nx.DiGraph()
    G.add_edges_from(edge_index.t().cpu().numpy())
    # 创建线图
    LG = nx.line_graph(G)
    # 将线图转换为PyG数据对象
    node_mapping = {node: idx for idx, node in enumerate(G.edges())}

    line_graph_edges = [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in LG.edges()]

    line_graph_edge_index = torch.tensor(line_graph_edges, dtype=torch.long, device=device).t().contiguous()

    next_label = 0
    label_combinations = {}     # 标签组合
    line_graph_labels = []
    for edge in LG.nodes():
        source, target = edge
        label_pair = tuple(sorted((data.y[source].item(), data.y[target].item())))
        if label_pair not in label_combinations:
            label_combinations[label_pair] = next_label
            next_label += 1
        line_graph_labels.append(label_combinations[label_pair])

    line_graph_data = Data(x=torch.eye(num_edges, device=device), edge_index=line_graph_edge_index)
    line_graph_data.y = torch.tensor(line_graph_labels, dtype=torch.long, device=device)
    # 创建新的mask
    # line_graph_data.train_mask = data.train_mask[edge_index[0]] & data.train_mask[edge_index[1]]
    # line_graph_data.val_mask = data.val_mask[edge_index[0]] & data.val_mask[edge_index[1]]
    # line_graph_data.test_mask = data.test_mask[edge_index[0]] & data.test_mask[edge_index[1]]

    return line_graph_data, line_graph_edge_index, label_combinations,node_mapping

def find_line_graph_subgraph(edge_to_idx, line_graph_edge_index, target_node):
    # 找到原始图中与目标节点相连的所有边
    original_graph_edges = [edge for edge in edge_to_idx.keys() if target_node in edge]

    # 将这些边转换为线图中的节点
    line_graph_nodes = [edge_to_idx[edge] for edge in original_graph_edges]

    # 找到线图中与这些节点相连的所有节点
    line_graph_subgraph_nodes = [node for node in range(line_graph_edge_index.size(1)) if set(line_graph_edge_index[:, node].tolist()).intersection(line_graph_nodes)]

    return line_graph_subgraph_nodes
################################ 构建子图 ######################################


def get_target_node_score(data, model, target_node):
    """
        function： 获取当前节点的得分,softmax得分
        args:      data: 全部数据集，data必须是BATCH类型
                   model: 模型
                   target_node: 目标节点的索引 type（int)
    """
    with torch.no_grad():
        model.eval()
        predicted_logits = model(data.x, data.edge_index, data.batch)
        predicted_label = predicted_logits[target_node].argmax().item() # 预测的标签
        predicted_probs = F.softmax(predicted_logits, dim=1)
        predicted_score = predicted_probs[target_node]
    return predicted_score[predicted_label]

def get_target_node_label(data, model, target_node):
    """
        function： 获取当前节点的预测标签
        args:      data: 全部数据集，data必须是BATCH类型
                   model: 模型
                   target_node: 目标节点的索引 type（int)
    """
    with torch.no_grad():
        model.eval()
        predicted_logits = model(data.x, data.edge_index)
        predicted_label = predicted_logits[target_node].argmax().item() # 预测的标签
    return predicted_label


def subgraph_to_line_graph3(subgraph_data, target_node, plot=True):
    """
    子图转线图

    Args:
        subgraph_data: 由k_hop_subgraph函数生成，然后转成的Data类型
        target_node:
        plot: 是否画图

    Returns:

    """
    edge_index = subgraph_data.edge_index
    num_edges = edge_index.size(1)
    device = edge_index.device
    edge_to_idx = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(num_edges)}

    line_graph_edges = []
    for i in range(num_edges):
        source, target = edge_index[:, i]
        for neighbor in edge_index[1, edge_index[0] == target]:
            if (source.item(), neighbor.item()) in edge_to_idx:
                line_graph_edges.append((i, edge_to_idx[(source.item(), neighbor.item())]))

    line_graph_edge_index = torch.tensor(line_graph_edges, dtype=torch.long, device=device).t().contiguous()
    line_graph_data = Data(x=torch.eye(num_edges, device=device), edge_index=line_graph_edge_index)

    line_graph_data.original_edge_info = edge_index

    target_edges_indices = []
    for i in range(num_edges):
        if target_node in edge_index[:, i]:
            print(i.device)
            # target_edges_indices.append(i)

    target_subgraph_edge_index = line_graph_edge_index[:, target_edges_indices]
    target_subgraph_data = Data(x=line_graph_data.x[target_edges_indices], edge_index=target_subgraph_edge_index)

    if plot:
        G = nx.Graph()
        for i in range(num_edges):
            G.add_node(i)
        for (src, dest) in line_graph_edges:
            G.add_edge(src, dest)

        node_colors = ['red' if node in target_edges_indices else 'gray' for node in G.nodes()]
        edge_colors = ['blue' for edge in G.edges()]

        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=700, font_weight='bold')
        plt.title("Line Graph")
        plt.show()

    return line_graph_data, target_subgraph_data



def visualize_subgraph(subgraph_data, target_node):
    """
    把子图画出来，目标节点标红

    Args:
        subgraph_data: 同上
        target_node:

    Returns:

    """

    # Create graph from subgraph edges
    G = nx.DiGraph()
    G.add_edges_from(subgraph_data.edge_index.T.cpu().numpy())

    # Generate node colors
    node_colors = ["red" if node == target_node else "blue" for node in G.nodes]

    # Draw graph with original node numbers
    nx.draw(G, with_labels=True, labels={node: node for node in G.nodes()}, node_color=node_colors)
    plt.show()
    plt.savefig('./a.png')


#################################### 博弈交互的计算 ####################################

def get_target_node_index(target_node,subset):
    """
    返回目标节点的下标

    Args:
        target_node:
        subset: 用在生成子图的节点列表中

    Returns:

    """
    return torch.nonzero(subset == target_node).squeeze()


def edge_shapley_value(data, model, target_node, src, dst, k=2):
    """
    计算边的夏普利值

    Args:
        data: 原始图数据
        model:
        target_node: 目标节点 tensor([1.]) 一这里是1纬的tensor类型
        src: 边的起点
        dst: 边的终点
        k: 目标节点的k跳范围

    Returns: (src,dst)这条边的shapley值

    """
    # 获取k跳子图及节点和边
    subset, edge_index, _, _ = k_hop_subgraph(
        target_node, k, data.edge_index, relabel_nodes=True)
    subgraph_node_features = data.x[subset]

    # 保存边的夏普利值
    shapley_value = 0.0

    # 移除当前边后的边集
    remaining_edge_index = edge_index[:, (edge_index[0] != src) | (edge_index[1] != dst)]

    target_node_index = get_target_node_index(target_node,subset)
    # 计算完整子图的目标值
    with torch.no_grad():
        target_logits = model(subgraph_node_features, edge_index)
        label = get_target_node_label(data,model,target_node)
        target_value = target_logits[target_node_index][label]

    # 计算移除当前边后的子图的目标值
    with torch.no_grad():
        subset_logits = model(subgraph_node_features, remaining_edge_index)
        subset_value = subset_logits[target_node_index][label]

    # 计算夏普利值
    for subset_size in range(1, remaining_edge_index.shape[1] + 1):
        for subset_indices in itertools.combinations(range(remaining_edge_index.shape[1]), subset_size):
            subset_edge_index = remaining_edge_index[:, list(subset_indices)]

            with torch.no_grad():
                subset_logits = model(subgraph_node_features, subset_edge_index)
                subset_value = subset_logits[target_node_index][label]

            shapley_value += (target_value - subset_value) / len(subset_indices)

    return shapley_value



def alliance_shapley_value(data, model, target_node, alliance_edges, k=2):
    """
    计算联盟的夏普利值

    Args:
        data: 原始图数据
        model: 模型
        target_node: 目标节点
        alliance_edges: 联盟的边（已经在联盟中的边），集合或者列表都行
        k: 目标节点的k跳范围

    Returns: 联盟的 Shapley 值

    """
    # 获取k跳子图及节点和边
    subset, edge_index, _, _ = k_hop_subgraph(
        target_node, k, data.edge_index, relabel_nodes=True)
    subgraph_node_features = data.x[subset]

    # 保存联盟的夏普利值
    shapley_value = 0.0

    # 移除联盟的边后的边集
    remaining_edge_index = edge_index

    for edge in alliance_edges:
        src, dst = edge
        remaining_edge_index = remaining_edge_index[:, (remaining_edge_index[0] != src) | (remaining_edge_index[1] != dst)]

    target_node_index = get_target_node_index(target_node, subset)

    # 计算完整子图的目标值
    with torch.no_grad():
        target_logits = model(subgraph_node_features, edge_index)
        label = get_target_node_label(data, model, target_node)
        target_value = target_logits[target_node_index][label]

    # 计算夏普利值
    for subset_size in range(1, len(alliance_edges) + 1):
        for subset_indices in itertools.combinations(range(len(alliance_edges)), subset_size):
            subset_edge_indices = list(subset_indices)
            subset_edge_index = remaining_edge_index[:, subset_edge_indices]

            with torch.no_grad():
                subset_logits = model(subgraph_node_features, subset_edge_index)
                subset_value = subset_logits[target_node_index][label]

            shapley_value += (target_value - subset_value) / len(subset_indices)

    return shapley_value

def shapley_interactions(data,model,target_node, allinance_edges, k = 2):
    # 获取k跳子图及节点和边

    m_payoff = 0
    for edge in allinance_edges:
        src, dst = edge
        m_payoff += edge_shapley_value(data,model,target_node,src,dst)

    return alliance_shapley_value(data,model,target_node,allinance_edges) - m_payoff


def find_max_shapley_edge(data,model,edge_index, alliance_edges, target_node,subset,threshold = 0.2):
    """
    获取新的具有最大联盟夏普利值的
    要注意，使用这个之前需要初始化空集，要不然报错

    Args:
        edge_index: 边的索引
        alliance_edges: 联盟中边的索引
        target_node: 目标节点

    Returns: 下一个边

    """
    max_shapley_value = -1
    max_shapley_edges = []

    # 初始化列表
    if not alliance_edges:
        candidate_edges = find_edges_starting_from_target(edge_index, target_node, subset)  # Replace with your edge-finding function
        for src, dst in candidate_edges:
            # Calculate the Shapley value for each candidate edge
            shapley_value = edge_shapley_value(data, model, target_node, src, dst, k=2)  # 请确保有相应的函数来计算Shapley值

            if shapley_value > max_shapley_value:
                max_shapley_value = shapley_value
                max_shapley_edges = [(src, dst)]
            elif abs(shapley_value - max_shapley_value) <= threshold:
                max_shapley_edges.append((src, dst))
        return max_shapley_edges
    else:
        for src, dst in edge_index.t().tolist():
            edge = (src, dst)
            if edge not in alliance_edges:
                # Create a new alliance by adding the edge to the existing ones
                new_alliance = list(alliance_edges) + [edge]
                if is_alliance_connected(new_alliance):  # 检查新的联盟是否连通
                    # Calculate the Shapley value for the new alliance
                    shapley_value = shapley_interactions(data,model,target_node,new_alliance) # 请确保有相应的函数来计算Shapley值
                    # If the new Shapley value is greater, update the maximum
                    if shapley_value > max_shapley_value:
                        max_shapley_value = shapley_value
                        max_shapley_edges = [edge]
                    elif abs(shapley_value - max_shapley_value) <= threshold:
                        max_shapley_edges.append(edge)

        return max_shapley_edges



def find_edges_starting_from_target(edge_index, target_node, subset):
    """
    初始化联盟集合的时候，找到以目标节点为起点的边

    Args:
        edge_index: 子图的边
        target_node: 目标节点

    Returns:

    """
    edges_starting_from_target = []
    target_node_index = torch.nonzero(subset == target_node).squeeze()

    for src, dst in edge_index.t().tolist():
        if src == target_node_index:
            edges_starting_from_target.append((src, dst))

    return edges_starting_from_target

def find_adjacent_edges(edge_index,target_node):

    edges_starting_from_target = []
    # target_node = target_node.item()

    for src, dst in edge_index.t().tolist():
        if src == target_node:
            edges_starting_from_target.append((src, dst))

    return edges_starting_from_target


def is_alliance_connected(alliance_edges):
    """
    判断是否是连通图，如果不连通就剪枝

    Args:
        alliance_edges:

    Returns:

    """
    alliance_graph = nx.Graph(alliance_edges)
    return nx.is_connected(alliance_graph)


def fidelity_score(model, data, subgraph_data, target_node, mask_edges_index):

    mask_edges_index = torch.tensor(mask_edges_index, dtype=torch.long).t().contiguous()
    model.eval()
    with torch.no_grad():
        # Original prediction
        logits_original = model(data.x, data.edge_index)
        probs_original = torch.softmax(logits_original, dim=1)
        label_original = torch.argmax(probs_original[target_node]).item()
        score_original = probs_original[target_node][0][label_original].item()

        # Prediction with masked edges
        logits_masked = model(data.x, mask_edges_index)
        # softmax
        probs_masked = torch.softmax(logits_masked, dim=1)
        label_masked = torch.argmax(probs_masked[target_node]).item()
        score_masked = probs_masked[target_node][0][label_original].item()

    fidelity = abs(score_original - score_masked)
    fidelity_pro = abs(score_original - score_masked) / score_original
    return fidelity, fidelity_pro



def fidelity_inv_score(model, data, target_node_index, mask_edges_index):
    mask_edges_index = torch.tensor(mask_edges_index, dtype=torch.long).t().contiguous()
    model.eval()
    G_original = nx.Graph()
    G_original.add_edges_from(data.edge_index.t().cpu().numpy())

    G_without_important_edges = G_original.copy()
    G_without_important_edges.remove_edges_from(mask_edges_index)

    edge_index_without_important_edges = torch.tensor(list(G_without_important_edges.edges), dtype=torch.long).t().contiguous()
    data_without_important_edges = Data(x=data.x, edge_index=edge_index_without_important_edges)


    with torch.no_grad():
        # Original prediction
        logits_original = model(data.x, data.edge_index)
        probs_original = torch.softmax(logits_original, dim=1)
        label_original = torch.argmax(probs_original[target_node_index]).item()
        score_original = probs_original[target_node_index][label_original].item()

        # Prediction with masked edges
        # Assuming mask_edges_index contains the masked edges and their indices
        logits_masked = model(data.x, data_without_important_edges.edge_index)
        probs_masked = torch.softmax(logits_masked, dim=1)
        label_masked = torch.argmax(probs_masked[target_node_index]).item()
        score_masked = probs_masked[target_node_index][label_masked].item()

    fidelity = abs(score_original - score_masked)
    fidelity_pro = abs(score_original - score_masked)/score_original
    return fidelity


def sparsity_score(alliance_edges, subgraph_org_data, data):
    num_all_edges = data.edge_index.size(1)
    num_edges = subgraph_org_data.edge_index.size(1)
    selected_edges = torch.tensor(alliance_edges, dtype=torch.long).t().contiguous()
    selected_edge_count = selected_edges.size(1)
    sparsity_score = 1.0 - (selected_edge_count / num_edges)
    sparsity_score_all = 1.0 - (selected_edge_count / num_all_edges)
    return sparsity_score, sparsity_score_all


def plot():
    G = nx.DiGraph()
    G.add_edges_from(subgraph_data.edge_index.T.cpu().numpy())

    edges = alliance
    edges = [(int(u), int(v)) for u, v in edges]
    H = nx.DiGraph()
    H.add_edges_from(edges)

    edge_colors = ['red' if G.has_edge(u, v) and H.has_edge(u, v) else 'gray' for u, v in G.edges()]
    edge_widths = [2 if edge in H.edges() else 1 for edge in G.edges()]

    node_colors = ['red' if node == target_node_index else "blue" for node in G.nodes()]
    node_sizes = [500 if node == target_node_index else 300 for node in G.nodes()]
    # 标题

    # 展示标题
    # 保留两位小数
    fig, ax = plt.subplots(figsize=(10, 10))

    # 在子图上绘制网络图
    nx.draw(G, with_labels=True, edge_color=edge_colors, node_color=node_colors, node_size=node_sizes,
        width=edge_widths, ax=ax)

    # 添加标题
    ax.set_title('node{}, fidelity:{:.3f}, sparsity: {:.3f}'.format(target_node.item(), fidelity, sparsity))

    plt.figure(figsize=(10, 10))
    plt.show()

# 计算阶乘
def jiecheng(a):
    if a == 0 or a == 1:
        return 1
    else:
        return a*jiecheng(a-1)


def save_node_explain_result(save_pth, file_name, explain_result):
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    file_path = os.path.join(save_pth, file_name+'.json')

    if os.path.exists(file_path):
        # 读取现有数据
        with open(file_path, 'r') as f:
            results = json.load(f)
    else:
        results = []

    results.append(explain_result)
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def save_all_explain_result(save_pth, file_name, explain_result):

    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    file_path = os.path.join(save_pth, file_name + '.json')

    if os.path.exists(file_path):
        # 读取现有数据
        with open(file_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
    results.append(explain_result)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)