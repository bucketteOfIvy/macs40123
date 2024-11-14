# This script is adapted from the class example and the Pytorch geometric 
# example here: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import pickle
import sys
import geopandas as gpd
import torch
#from sklearn.manifold import TSNE
from osmnx import graph_from_gdfs
from torch_geometric.nn import Node2Vec
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from torch_geometric.data import Data
from collections import defaultdict
from torch import Tensor

def from_osmnx(G, nodes_gdf, edges_gdf, group_node_attrs=None, group_edge_attrs=None):
    '''
    Converts osmnx graph or digraph to torch_geometric.data.Data objects.
    Based nearly entirely on torch.geometry.utils.convert.from_networkx, 
    but with modifications for this specific project.
    Args:
      G (networkx.Graph or networx.DiGraph): A networkx graph
      nodes_gdf (GeoDataFrame): the initial gdf of nodes.
      edges_gdf (GeoDataFrame)L the initial gdf of edges.
      group_node_attrs (List[str], "all", or None): The node attributes to be
        concatenated and added to data.x (defaults to None)
      group_edge_attrs (List[str], "all", or None): The edge attributes to be
        concatenated and added to data.edge_attr. Defaults to None.
       All attributes must be numeric (woe)
    '''
    
    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict = defaultdict(list)
    data_dict['edge_index'] = edge_index

    node_attrs = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    
    edge_attrs = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    
    if group_node_attrs and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    # Main change from the initial from_networkx function are these
    # two chunks. Instead of raising an error, they now just 
    # reference my initial GDFs, since I can do that. 
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            feat_dict = {}
            print(f"Correcting dropped node values for row {i}")
            for k in node_attrs:
                feat_dict[k] = nodes_gdf.iloc[i].get(k)
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)
    
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):        
        if set(feat_dict.keys()) != set(edge_attrs):
            feat_dict = {}
            print(f"Correcting dropped edge values for row {i}")
            for k in edge_attrs:
                feat_dict[k] = edges_gdf.iloc[i].get(k)
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data_dict[str(key)].append(value)
    
    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue
        key = f'graph_{key}' if key in node_attrs else key
        data_dict[str(key)] = value
   
    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = torch.stack(value, dim=0)
        else:
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception:
                pass
    
    data = Data.from_dict(data_dict)

    if group_node_attrs:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)
    
    if group_edge_attrs:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)
    
    if data.x is not None and data.pos is not None:
        data.num_nodes = G.number_of_nodes()
   
    return data
        

def setup_model(data):
    '''
    Setup our model. Returns model, loader, optimizer, device as a tuple.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True
    ).to(device)

    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=129, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    return model, loader, optimizer, device

def read_in_graphs(edges_loc="../../data/shapes/newyork_edges_2020.gpkg",
                   nodes_loc="../../data/shapes/newyork_nodes_2020_split.gpkg"):
    '''
    Read in graph data, using assumed file names and locations, and return
    both the pytorch graph and a dataset mapping from GEOIDs to graph indices.
    '''
    edges = gpd.read_file(edges_loc, engine='pyogrio', use_arrow=True)
    edges = edges.set_index(['u', 'v', 'key'])
    nodes = gpd.read_file(nodes_loc, engine='pyogrio', use_arrow=True)
    nodes = nodes.set_index(['osmid'])
    G = graph_from_gdfs(nodes[['y', 'x']], edges[['geometry']])

    # We need this in pytorch's graph format
    data = from_osmnx(G, nodes, edges)

    # make dict
    # above process was order preserving
    geoid_dict = {}
    for i, val in enumerate(nodes.GEOID.tolist()):
        if val not in geoid_dict:
            geoid_dict[val] = [i]
            continue
        geoid_dict[val].append(i)

    # No longer need this, and do not want
    # it considered in node2vec
    del data.GEOID

    return data, geoid_dict

def train(model, loader, optimizer, device):
    '''
    Train node2vec model
    '''
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150
    )
    return acc

if __name__ == "__main__":

    # Setup
    print("Reading in data..")
    data, geoid_dict = read_in_graphs()

    print("Setting up model..")
    model, loader, optimizer, device = setup_model(data)

    # Training
    print("Training...")
    for epoch in range(1, 101):
        loss = train(model, loader, optimizer, device)
        acc = test(model, data)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Save model
    print("Saving..")
    torch.save(model, "../../data/models/node2vec_no_att")

    with open('../../data/models/geoid_to_index.pkl', 'wb') as f:
        pickle.dump(geoid_dict, f)

