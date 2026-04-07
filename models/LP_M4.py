#!/home/schoenstein/.conda/envs/graphe/bin/python
#SBATCH --job-name=LP_M4
#SBATCH --output=/home/schoenstein/these/slurm_out/slurm-%J.out --error=/home/schoenstein/these/slurm_out/slurm-%J.err



import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from datetime import datetime
import torch.nn as nn



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



train_data.x = torch.ones((train_data.num_nodes, 1))
val_data.x = train_data.x.clone()
test_data.x = train_data.x.clone()



train_data = train_data.to(device)
val_data = val_data.to(device)
#test_data = test_data.to(device)



model_n2v = Node2Vec(
    edge_index = train_data.edge_index,
    embedding_dim = settings["options"]["embedding_dim"],
    walk_length = settings["options"]["walk_length"],
    context_size = settings["options"]["context_size"],
    walks_per_node = settings["options"]["walks_per_node"],
    p = settings["options"]["p"],
    q = settings["options"]["q"],
    num_negative_samples = settings["options"]["num_negative_samples"],
    sparse=True
).to(device)
optimizer = torch.optim.SparseAdam(list(model_n2v.parameters()), lr=settings["options"]["lr"])
n2v_loader = model_n2v.loader(batch_size = 128, shuffle =  True)



def train_epoch():
    model_n2v.train()
    total_loss = 0
    count = 0
    for pos, neg in n2v_loader:
        pos = pos.to(device)
        neg = neg.to(device)
        optimizer.zero_grad()
        loss = model_n2v.loss(pos, neg)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        count = count + 1
    return total_loss / count
now = datetime.now()
output_path = settings["output"] + "model_LP_M4_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "_E" + str(epoch) + ".pth"
with open(output_path, "a") as output:
    for epoch in range(1, 10):
        loss = train_epoch()
        output.write(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    output_model = settings["output"] + "model_LP_M4_n2v_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ".pth"
    torch.save(model_n2v.state_dict(), output_model)
    model_n2v.eval()
    z = model_n2v().to(device)



class Predictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_label_index):
        edge_emb_src = x[edge_label_index[0]]
        edge_emb_dst = x[edge_label_index[1]]
        edge_emb = torch.cat([edge_emb_src, edge_emb_dst], dim=-1)
        return self.mlp(edge_emb).view(-1)

predictor = Predictor(hidden_channels=settings["options"]["hidden_channels"]).to(device)



optimizer_mlp = torch.optim.Adam(predictor.parameters(), lr=settings["options"]["lr"])



def train_epoch2():
    predictor.train()
    optimizer_mlp.zero_grad()
    pred = predictor(z, train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(pred, train_data.edge_label.float())
    loss.backward()
    optimizer_mlp.step()
    return loss
def evaluate():
    predictor.eval()
    y_pred = predictor(z, val_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(y_pred, val_data.edge_label.float()).item()
    auc = roc_auc_score(val_data.edge_label.cpu().numpy(), y_pred.detach().cpu().numpy())
    ap = average_precision_score(val_data.edge_label.cpu().numpy(), y_pred.detach().cpu().numpy())
    return loss, auc, ap



with open(output_path, "a") as output:
    best_val_auc = 0
    limit = settings["options"]["early_stop"]
    count = 0
    val_aps = []
    val_aucs = []
    for epoch in range(1, 50):
        loss = train_epoch2()
        val_loss, val_auc, val_ap = evaluate()
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Loss : {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
        if val_auc > best_val_auc:
            output_model = settings["output"] + "model_LP_M4_" + str(now.day) + "-" + str(now.month) + "-" + str(now.year) + "_" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second) + "_E" + str(epoch) + ".pth"
            torch.save(predictor.state_dict(), output_model)
            best_val_auc = val_auc
            count = 0
        else:
            count =  count + 1
            if count >= limit:
                print("Early stop")
                break
    output.write("\n")
    output.write("\n")
    output.write("\n")
    json.dump(settings, output, indent = 2)