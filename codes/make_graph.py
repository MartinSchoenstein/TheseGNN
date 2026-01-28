import pandas as pd
import networkx as nx
import sys

def ortho_graph_(path_taxid):

    #Importe liste des espèces
    id_list = []
    with open(path_taxid) as id_file:
        for l in id_file:
            id_list.append(int(l.strip()))
    print("Chargement : ", len(id_list), " espèces")

    #Importe fichiers d'orthologies
    df_ortho = pd.DataFrame(columns=["taxid", "protein/inparalogs", "orthologs", "ortholog_taxid"])
    for id in id_list:
        df = pd.read_csv("/home/schoenstein/these/data/" + str(id) + ".tsv", sep = "\t")
        df = df[df['ortholog_taxid'].isin(id_list)]
        df["taxid"] = id
        df_ortho = pd.concat([df_ortho, df], ignore_index = True)
    
    #Etale les paralogues
    cols_to_explode = ['protein/inparalogs', 'orthologs']
    for col in cols_to_explode:
        df_ortho[col] = df_ortho[col].str.split(',')
    df_ortho_split = df_ortho.copy()
    for col in cols_to_explode:
        df_ortho_split = df_ortho_split.explode(col, ignore_index=True)
    print("Chargement : ", len(df_ortho_split), " relations")

    #Crée graphe
    edges_ortho = list(zip(df_ortho_split['protein/inparalogs'], df_ortho_split['orthologs']))
    G = nx.Graph()
    G.add_edges_from(edges_ortho)

    #Attribue les espèces à chaque noeud
    if len(sys.argv) > 3:
        degree = G.degree
        max_degree = max(dict(degree).values())
        print(max_degree)
        degree_norm = {node : deg/max_degree for node, deg in dict(degree).items()}
        cluster_coef = nx.clustering(G)
        nx.set_node_attributes(G, degree_norm, name='degree')
        nx.set_node_attributes(G, cluster_coef, name='clustering')
        print("Attributs sur noeuds : degrées et coefficient de clustering")                  
    #Composantes connexes et leurs tailles
    cc = sorted(list(nx.connected_components(G)), key = len, reverse = True)
    list_taille = []
    for c in cc:
        list_taille.append(len(c))

    #Statistiques de base du graph
    print("Nombre de nœuds : ", G.number_of_nodes())
    print("Nombre d’arêtes : ", G.number_of_edges())
    print("Nombre de composantes connexes : ", len(cc))
    print(f"Composante la plus grande : ", len(max(cc, key=len)))

    return G


G = ortho_graph_(sys.argv[1])

nx.write_gml(G, sys.argv[2])
