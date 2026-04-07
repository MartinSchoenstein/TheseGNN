#!/home/schoenstein/.conda/envs/graphe/bin/python
#SBATCH --job-name=LP_M3
#SBATCH --output=/home/schoenstein/these/slurm_out/slurm-%J.out --error=/home/schoenstein/these/slurm_out/slurm-%J.err



import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from datetime import datetime
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



settings = {"graph": "/home/schoenstein/these/graph/graph_light.gml",
    "options": {
            "negative_sampling": "double split2",
            "negative_sampling_ratio": 0.5,              #1.5 our obtenir 0.8 en double split1
            "attributes": "statistics",
            "aggr": "lstm",
            "hidden_channels": 64,
            "num_neighbors": [25, 10],
            "batch_size": 128,
            "lr": 0.01,
            "early_stop": 6
        },
    "output": "/home/schoenstein/these/models/training/"
        }



G = nx.read_gml(settings["graph"])
node_to_idx = {}
for i, n in enumerate(G.nodes()):
    node_to_idx[n] = i
    G.nodes[n]['id'] = i
G = nx.relabel_nodes(G, node_to_idx)
data = from_networkx(G)



torch.manual_seed(42)



if settings["options"]["negative_sampling"] == "random":
    transform = T.RandomLinkSplit(
        num_val = 0.1,  
        num_test = 0.1,  
        disjoint_train_ratio = 0,  
        neg_sampling_ratio = 1,
        is_undirected = True
    )
    train_data, val_data, test_data = transform(data)
elif settings["options"]["negative_sampling"] == "double split1":
    cc = data.connected_components()
    neg_inside = 0
    neg_inside_val = 0
    train_list = []
    val_list = []
    test_list = []
    negative_sampling_ratio = settings["options"]["negative_sampling_ratio"]
    for c in cc:
        transform = T.RandomLinkSplit(
                num_val = 0.1,  
                num_test = 0.1,  
                disjoint_train_ratio = 0,  
                neg_sampling_ratio = negative_sampling_ratio,
                is_undirected = True
            )
        train_data2, val_data2, test_data2 = transform(c)
        train_data2.edge_index = train_data2.id[train_data2.edge_index]
        test_data2.edge_index = test_data2.id[test_data2.edge_index]
        val_data2.edge_index = val_data2.id[val_data2.edge_index]
        train_data2.edge_label_index = train_data2.id[train_data2.edge_label_index]
        val_data2.edge_label_index = val_data2.id[val_data2.edge_label_index]
        test_data2.edge_label_index = test_data2.id[test_data2.edge_label_index]
        del train_data2.id
        del val_data2.id
        del test_data2.id
        neg_inside = neg_inside + (train_data2.edge_label == 0).sum().item()
        train_list.append(train_data2)
        val_list.append(val_data2)
        test_list.append(test_data2)
    ratio_neg_inside = neg_inside/(len(list(G.edges()))*2)
    print("Ratio neg intra cc : ", ratio_neg_inside)
    transform = T.RandomLinkSplit(
            num_val = 0.1,  
            num_test = 0.1,  
            disjoint_train_ratio = 0,  
            neg_sampling_ratio = 1 - ratio_neg_inside,
            is_undirected = True
        )
    train_data3, val_data3, test_data3 = transform(data)

    pos_train = train_data3.edge_label_index[:, train_data3.edge_label == 1]
    pos_val = val_data3.edge_label_index[:, val_data3.edge_label == 1]
    pos_test = test_data3.edge_label_index[:, test_data3.edge_label == 1]
    neg_train1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                             for d in train_list], dim=1) #neg intra cc
    neg_val1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                           for d in val_list], dim=1)
    neg_test1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                            for d in test_list], dim=1)
    neg_train2 = train_data3.edge_label_index[:, train_data3.edge_label==0]           
    neg_val2 = val_data3.edge_label_index[:, val_data3.edge_label==0]
    neg_test2 = test_data3.edge_label_index[:, test_data3.edge_label==0]
    neg_train = torch.cat([neg_train1, neg_train2], dim=1)
    neg_val = torch.cat([neg_val1, neg_val2], dim=1)
    neg_test = torch.cat([neg_test1, neg_test2], dim=1)
    train_data = Data(
        edge_index=train_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_train, neg_train], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_train.size(1), dtype=torch.long),
            torch.zeros(neg_train.size(1), dtype=torch.long)
        ])
    )
    val_data = Data(
        edge_index=val_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_val, neg_val], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_val.size(1), dtype=torch.long),
            torch.zeros(neg_val.size(1), dtype=torch.long)
        ])
    )
    test_data = Data(
        edge_index=test_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_test, neg_test], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_test.size(1), dtype=torch.long),
            torch.zeros(neg_test.size(1), dtype=torch.long)
        ])
    )
elif settings["options"]["negative_sampling"] == "double split2":
    cc = data.connected_components()
    neg_inside = 0
    neg_inside_val = 0
    train_list = []
    val_list = []
    test_list = []
    negative_sampling_ratio = settings["options"]["negative_sampling_ratio"]
    print("negative_sampling_ratio : ", negative_sampling_ratio)
    for c in cc:
        num_nodes = int(c.num_nodes)
        num_edges = int(c.num_edges)
        if (num_nodes * (num_nodes -1))/2 - num_edges/2 > num_edges/2 * negative_sampling_ratio:
            transform = T.RandomLinkSplit(
                    num_val = 0.1,  
                    num_test = 0.1,  
                    disjoint_train_ratio = 0,  
                    neg_sampling_ratio = negative_sampling_ratio,
                    is_undirected = True
                )
        else:
            transform = T.RandomLinkSplit(
                    num_val = 0.1,  
                    num_test = 0.1,  
                    disjoint_train_ratio = 0,  
                    neg_sampling_ratio = 0,
                    is_undirected = True
                )
        train_data2, val_data2, test_data2 = transform(c)
        train_data2.edge_index = train_data2.id[train_data2.edge_index]
        test_data2.edge_index = test_data2.id[test_data2.edge_index]
        val_data2.edge_index = val_data2.id[val_data2.edge_index]
        train_data2.edge_label_index = train_data2.id[train_data2.edge_label_index]
        val_data2.edge_label_index = val_data2.id[val_data2.edge_label_index]
        test_data2.edge_label_index = test_data2.id[test_data2.edge_label_index]
        del train_data2.id
        del val_data2.id
        del test_data2.id
        neg_inside = neg_inside + (train_data2.edge_label == 0).sum().item()
        neg_total_goal = int(neg_inside/negative_sampling_ratio)
        neg_inside_val = neg_inside_val + (val_data2.edge_label == 0).sum().item()
        neg_total_goal_val = int(neg_inside_val/negative_sampling_ratio)
        train_list.append(train_data2)
        val_list.append(val_data2)
        test_list.append(test_data2)
    ratio_neg_inside = neg_inside/(len(list(G.edges()))*2)
    print("Ratio neg intra cc : ", ratio_neg_inside)
    transform = T.RandomLinkSplit(
            num_val = 0.1,  
            num_test = 0.1,  
            disjoint_train_ratio = 0,  
            neg_sampling_ratio = 1 - negative_sampling_ratio,
            is_undirected = True
        )
    train_data3, val_data3, test_data3 = transform(data)

    pos_train = train_data3.edge_label_index[:, train_data3.edge_label == 1]
    n_pos_train = pos_train.size(1)
    perm = torch.randperm(n_pos_train)[:neg_total_goal]
    pos_train = pos_train[:, perm] #Sous selection des positives du train

    neg_train = train_data3.edge_label_index[:, train_data3.edge_label == 0]
    n_neg_train = neg_train.size(1)
    perm_neg = torch.randperm(n_neg_train)[:int(neg_total_goal*(1 - negative_sampling_ratio))]
    neg_train = neg_train[:, perm_neg] #Sous selection des negatives globales du train

    print(neg_inside, neg_train.size(1), pos_train.size(1), n_pos_train)

    pos_val = val_data3.edge_label_index[:, val_data3.edge_label == 1]
    n_pos_val = pos_val.size(1)
    perm_val = torch.randperm(n_pos_val)[:neg_total_goal_val]
    pos_val = pos_val[:, perm_val] #val

    neg_val = val_data3.edge_label_index[:, val_data3.edge_label == 0]
    n_neg_val = neg_val.size(1)
    perm_neg_val = torch.randperm(n_neg_val)[:int(neg_total_goal_val*(1 - negative_sampling_ratio))]
    neg_val = neg_val[:, perm_neg_val] #val

    pos_test = test_data3.edge_label_index[:, test_data3.edge_label == 1]
    neg_train1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                             for d in train_list], dim=1) #neg intra cc
    neg_val1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                           for d in val_list], dim=1)
    neg_test1 = torch.cat([d.edge_label_index[:, d.edge_label==0] 
                            for d in test_list], dim=1)

    neg_train2 = neg_train           
    neg_val2 = neg_val 
    neg_test2 = test_data3.edge_label_index[:, test_data3.edge_label==0]
    neg_train = torch.cat([neg_train1, neg_train2], dim=1)
    neg_val = torch.cat([neg_val1, neg_val2], dim=1)
    neg_test = torch.cat([neg_test1, neg_test2], dim=1)
    train_data = Data(
        edge_index=train_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_train, neg_train], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_train.size(1), dtype=torch.long),
            torch.zeros(neg_train.size(1), dtype=torch.long)
        ])
    )
    val_data = Data(
        edge_index=val_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_val, neg_val], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_val.size(1), dtype=torch.long),
            torch.zeros(neg_val.size(1), dtype=torch.long)
        ])
    )
    test_data = Data(
        edge_index=test_data3.edge_index,
        num_nodes=data.num_nodes,
        edge_label_index=torch.cat([pos_test, neg_test], dim=1),
        edge_label=torch.cat([
            torch.ones(pos_test.size(1), dtype=torch.long),
            torch.zeros(neg_test.size(1), dtype=torch.long)
        ])
    )


def labelling(batch):
    batch = batch.cpu()
    edge_index = batch.edge_index
    u, v = batch.edge_label_index
    mask = ~(((edge_index[0] == u) & (edge_index[1] == v)) | ((edge_index[0] == v) & (edge_index[1] == u)))
    edge_index = edge_index[:, mask]
    G = nx.Graph()
    G.add_nodes_from(range(batch.num_nodes))
    G.add_edges_from(edge_index.t().tolist())
    du = nx.single_source_shortest_path_length(G, int(u))
    dv = nx.single_source_shortest_path_length(G, int(v))
    labels = []
    for n in range(batch.num_nodes):
        if n == u or n == v:
            labels.append(1)
        else:
            dun = du.get(n, "unr")
            dvn = dv.get(n, "unr")
            if dun == "unr" or dvn == "unr":
                 labels.append(0)
            else:
                labels.append(1 + min(dun, dvn) + ((dun + dvn - 2)*(dun + dvn - 1)) // 2)
    return torch.tensor(labels).unsqueeze(-1).float(), edge_index  



if settings["options"]["attributes"] == "unique":
    train_data.x = torch.ones((train_data.num_nodes, 1))
    val_data.x = train_data.x.clone()
    test_data.x = train_data.x.clone()
elif settings["options"]["attributes"] == "statistics":
    G_train = nx.Graph()
    G_train.add_nodes_from(range(train_data.num_nodes))
    G_train.add_edges_from(train_data.edge_index.t().tolist())
    degree = dict(G_train.degree())
    max_degree = max(degree.values())
    degree_norm = {n: d/max_degree for n, d in degree.items()}
    clustering = nx.clustering(G_train)
    degree_tensor = torch.tensor(list(degree_norm.values()), dtype=torch.float32)
    clustering_tensor = torch.tensor(list(clustering.values()), dtype=torch.float32)
    train_data.x = torch.stack([degree_tensor, clustering_tensor], dim=-1)
    val_data.x = train_data.x.clone()
    test_data.x = train_data.x.clone()



#train_data = train_data.to(device)
#val_data = val_data.to(device)
#test_data = test_data.to(device)



train_loader = LinkNeighborLoader(
    data = train_data,
    num_neighbors = settings["options"]["num_neighbors"],
    edge_label_index = train_data.edge_label_index,
    edge_label = train_data.edge_label,
    subgraph_type = "induced",
    batch_size = settings["options"]["batch_size"],
    shuffle = True
)
val_loader = LinkNeighborLoader(
    data = val_data,
    num_neighbors = settings["options"]["num_neighbors"], 
    edge_label_index = val_data.edge_label_index,
    edge_label = val_data.edge_label,
    subgraph_type = "induced",
    batch_size = settings["options"]["batch_size"],
    shuffle = True
)



if settings["options"]["attributes"] == "unique":
    train_subgraphs_data = []
    for batch in train_loader:
        batch.x, batch.edge_index = labelling(batch)
        data = Data(
            x = batch.x,
            edge_index = batch.edge_index,  
            edge_label = batch.edge_label,      
            edge_label_index = batch.edge_label_index  
        )
        train_subgraphs_data.append(data)
    val_subgraphs_data = []
    for batch in val_loader:
        batch.x, batch.edge_index = labelling(batch)
        data = Data(
            x = batch.x,
            edge_index = batch.edge_index,  
            edge_label = batch.edge_label,      
            edge_label_index = batch.edge_label_index  
        )
        val_subgraphs_data.append(data)
elif settings["options"]["attributes"] == "statistics":
    train_subgraphs_data = []
    for batch in train_loader:
        x_label, batch.edge_index = labelling(batch)
        x_stats = batch.x
        batch.x = torch.cat([x_label, x_stats], dim = 1)
        data = Data(
            x = batch.x,
            edge_index = batch.edge_index,  
            edge_label = batch.edge_label,      
            edge_label_index = batch.edge_label_index  
        )
        train_subgraphs_data.append(data)
    val_subgraphs_data = []
    for batch in val_loader:
        x_label, batch.edge_index = labelling(batch)
        x_stats = batch.x
        batch.x = torch.cat([x_label, x_stats], dim = 1)
        data = Data(
            x = batch.x,
            edge_index = batch.edge_index,  
            edge_label = batch.edge_label,      
            edge_label_index = batch.edge_label_index  
        )
        val_subgraphs_data.append(data)




class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
class Predictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
    def forward(self, x):
        return self.mlp(x).view(-1)
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gnn = GNN(in_channels, hidden_channels)
        self.predictor = Predictor(hidden_channels)
    def forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        pred = self.predictor(x)
        return pred
if settings["options"]["attributes"] == "statistics":
    model = Model(in_channels = 3, hidden_channels = settings["options"]["hidden_channels"]).to(device)
if settings["options"]["attributes"] == "unique":
    model = Model(in_channels = 1, hidden_channels = settings["options"]["hidden_channels"]).to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=settings["options"]["lr"])



train_reloader = DataLoader(train_subgraphs_data, batch_size=settings["options"]["batch_size2"], shuffle=True)
val_reloader = DataLoader(val_subgraphs_data, batch_size=settings["options"]["batch_size2"])
                          


def train_epoch():
    model.train()
    total_loss = 0
    count = 0
    for batch in train_reloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        count = count + 1
    return total_loss / count
def evaluate():
    model.eval()
    y_truth = []
    y_pred = []
    total_loss = 0
    count = 0
    for batch in val_reloader:
        batch = batch.to(device)
        pred = model(batch)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label.float())
        y_truth.append(batch.edge_label.cpu())
        y_pred.append(torch.sigmoid(pred).cpu())
        total_loss = total_loss + loss.item()
        count = count + 1
    y_truth = torch.cat(y_truth).cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    auc = roc_auc_score(y_truth, y_pred)
    ap = average_precision_score(y_truth, y_pred)
    return total_loss / count, auc, ap



now = datetime.now()
output_path = settings["output"] + "LP_M3_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + ".txt"
with open(output_path, "w") as output:
    best_val_auc = 0
    limit = settings["options"]["early_stop"]
    count = 0
    train_losses = []
    val_aps = []
    val_aucs = []
    for epoch in range(1, 50):
        loss = train_epoch()
        train_losses.append(loss)
        val_loss, val_auc, val_ap = evaluate()
        val_aps.append(val_ap)
        val_aucs.append(val_auc)
        if val_auc > best_val_auc:
            output_model = settings["output"] + "model_LP_M3_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "_E" + str(epoch) + ".pth"
            torch.save(model.state_dict(), output_model)
            best_val_auc = val_auc
            count = 0
        else:
            count =  count + 1
            if count >= limit:
                output.write("Early stop")
                break
    output.write("\n")
    output.write("\n")
    output.write("\n")
    json.dump(settings, output, indent = 2)