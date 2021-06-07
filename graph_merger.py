import networkx as nx

def filename_to_label(filename):
    """Given a filename, returns a label to be added to nodes indices"""
    
    node_label = filename.split('.')[0].split('/')[1]

    return node_label

def merge_graphs(input_filenames, output_filename):

    graphs = []
    graphs_names = []
    for filename in input_filenames:
        graphs.append(nx.read_gml(filename))
        graphs_names.append(f'{filename_to_label(filename)}-')

    graph_union = nx.union_all(graphs, rename=graphs_names)   
    nx.write_gml(graph_union, output_filename)