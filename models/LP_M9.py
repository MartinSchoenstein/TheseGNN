#!/home/schoenstein/.conda/envs/graphe/bin/python
#SBATCH --job-name=LP_M9
#SBATCH --output=/home/schoenstein/these/slurm_out/slurm-%J.out --error=/home/schoenstein/these/slurm_out/slurm-%J.err



import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GAE
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from datetime import datetime
from torch_geometric.nn import GCNConv



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with open("M2_settings.json", "r") as file:
    settings = json.load(file)



G = nx.read_gml(settings["graph"])
data = from_networkx(G)



if settings["options"]["negative_sampling"] == "random":
    transform = T.RandomLinkSplit(
        num_val = 0.1,  
        num_test = 0.1,  
        disjoint_train_ratio = 0,  
        neg_sampling_ratio = 1,
        is_undirected = True
    )
    train_data, val_data, test_data = transform(data)

elif settings["options"]["negative_sampling"] == "double split":
    cc = list(nx.connected_components(G))
    neg_inside = 0
    train_list = []
    val_list = []
    test_list = []
    for c in cc:
        G2 = G.subgraph(c).copy()
        data2 = from_networkx(G2)
        transform = T.RandomLinkSplit(
                num_val = 0.1,  
                num_test = 0.1,  
                disjoint_train_ratio = 0,  
                neg_sampling_ratio = 1.5,
                is_undirected = True
            )
        train_data2, val_data2, test_data2 = transform(data2)
        neg_inside = neg_inside + len(train_data2.edge_label)
        train_list.append(train_data2)
        val_list.append(val_data2)
        test_list.append(test_data2)
    ratio_neg_inside = neg_inside/(len(list(G.edges()))*2)
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
                             for d in train_list], dim=1)
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



train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)



class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)
model = GAE(encoder=Encoder(in_channels = train_data.x.shape[1], hidden_channels = settings["options"]["hidden_channels"]))
optimizer = torch.optim.Adam(model.parameters(), lr=settings["options"]["lr"])



def train_epoch():
    model.train()
    optimizer.zero_grad()
    pred = model.encode(train_data.x, train_data.edge_index)
    pos_edge_index = val_data.edge_label_index[:, val_data.edge_label==1]
    neg_edge_index = val_data.edge_label_index[:, val_data.edge_label==0]
    loss = model.recon_loss(pred, pos_edge_index, neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()
def evaluate():
    model.eval()
    pred = model.encode(val_data.x, val_data.edge_index)
    pos_edge_index = val_data.edge_label_index[:, val_data.edge_label==1]
    neg_edge_index = val_data.edge_label_index[:, val_data.edge_label==0]
    loss = model.recon_loss(pred, pos_edge_index, neg_edge_index)
    auc, ap = model.test(pred, pos_edge_index, neg_edge_index)
    return loss, auc, ap



now = datetime.now()
output_path = settings["output"] + "LP_M9_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ".txt"
with open(output_path, "w") as output:
    best_val_auc = 0
    limit = settings["options"]["early_stop"]
    count = 0
    for epoch in range(1, 50):
        loss = train_epoch()
        val_loss, val_auc, val_ap = evaluate()
        output.write(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Loss : {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}\n")
        if val_auc > best_val_auc:
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