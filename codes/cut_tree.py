from ete3 import Tree

id_list = []
with open("/home/schoenstein/these/data/id_list_metazoa.txt") as id_file:
    for l in id_file:
        id_list.append(l.strip())

t = Tree("/home/schoenstein/these/data/Eukaryota2023.nwk", format = 1)
subtree = t.copy()
subtree.prune(id_list)
subtree.write(outfile="/home/schoenstein/these/gnn_postbiblio/Eukaryota2023_fullcut.nwk", format=1)