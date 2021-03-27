import numpy as np
from skimage.measure import label, regionprops
from scipy.stats import zscore
import networkx as nx
import misc

def get_shape_props_from_mask(img_mask, props_to_measure, connectivity=1, return_scikit_props=False):
    '''Return shape properties calculated for glands in image `img_mask`. `props_to_measure` is a list
    of shape properties name to calculate.'''
    
    label_img, qtt = label(img_mask, return_num=True, connectivity=connectivity)
    props = regionprops(label_img)
    all_shape_props = []
    for idx, prop in enumerate(props):
        shape_prop = []
        for prop_name in props_to_measure:
            shape_prop.append(prop[prop_name])
        all_shape_props.append(shape_prop)

    all_shape_props = np.array(all_shape_props)
    
    if return_scikit_props:
        # Also returns list from scikit-image containing RegionProperties objects
        return all_shape_props, props
    else:
        return all_shape_props

def get_graph_props(nxgraph):
    '''Return graph properties calculated for a networkx graph. The graph must have an attribute called 'weight'. '''
    
    degree = dict( nxgraph.degree() )
    strength = dict( nxgraph.degree(weight='weight') )
    betweenness = nx.betweenness_centrality(nxgraph, weight='weight')

    all_node_props = []
    for idx in range(len(nxgraph)):
        node_prop = []
        node_prop.append(degree[idx])
        node_prop.append(strength[idx])
        node_prop.append(betweenness[idx])
        all_node_props.append(node_prop)   

    all_node_props = np.array(all_node_props)
    
    return all_node_props

    
def display_shape_props(img_mask, props_to_measure, shape_label, connectivity=1):
    '''Show the gland corresponding to label `shape_label` and also some shape properties.'''
    
    all_shape_props, props = get_shape_props_from_mask(img_mask, props_to_measure, connectivity=connectivity, 
                                                       return_scikit_props=True)
    prop = props[shape_label-1]    
    # Get center of mass for the gland
    center = list(map(int, prop.centroid))
    # Show gland image
    misc.show_img(prop.image)
    print(f'Shape located at {center} has properties:')
    for prop_name in props_to_measure:
        print(f'{prop_name}: {prop[prop_name]:.2f}')
                
def normalize_values(vals, means=None, stds=None):
    '''Normalize values in list `vals` using the equation
    
    vals_norm = (vals-means)/stds
    
    The list can be one-dimensional or bi-dimensional (N rows representing objects and M columns 
    representing features). If `means` and `stds` are given, they are used for normalizing the values. 
    Otherwise, the mean and standard deviation are calculated from the values. 
    '''
    
    vals = np.array(vals)
    
    if (means is None) and (stds is None):
        vals = zscore(vals)
    else:
        vals = (vals - means)/stds
        
    return vals

def calculate_weight(pos_node1=None, pos_node2=None, att_node1=None, att_node2=None, alpha=0.):
    '''Calculate edge weight for a pair of nodes with the given positions and attributes. `alpha` adjusts
    the relative importance between position and attribute. If alpha=0, only attributes are used. If 
    alpha=1, only the position of the nodes are used.'''
    
    dist_pos2 = np.sum((pos_node1-pos_node2)**2)
    dist_att2 = np.sum((att_node1-att_node2)**2)
    
    dist2 = alpha*dist_pos2 + (1-alpha)*dist_att2
    weight = np.exp(-np.sqrt(dist2))
    return weight
    
def calculate_weight_all(nxgraph, pos_nodes, att_nodes, alpha=0., att_idx=None, normalize_pos=True, 
                         normalize_att=True, pos_means=None, pos_stds=None, att_means=None, att_stds=None):
    '''Calculate edge weights for all nodes in the graph. 
    
    Parameters
    ----------
    nxgraph : networkx graph
      Graph to calculate the weights
    pos_nodes : list
      Positions of the nodes in the graph
    att_nodes : list
      List of nodes attributes (e.g.: shape properties)
    alpha : float
      Relative importante of positions and attributes when calculating the weight (see calculate_weight())
    att_idx : int
        Attribute index to use for weight calculation. If None, all attributes are used.
    normalize_pos : bool
        Whether the positions should be normalized
    normalize_att : bool
        Whether the attributes should be normalized
    pos_means : list
        Averages calculated for the positions, used in the normalization. Must have length two
    pos_stds : list
        Standard deviations calculated for the positions, used in the normalization. Must have length two
    att_means : list
        Averages calculated for the attributed, used in the normalization. Must have length equal to
        the number of attributes
    att_stds : list
        Standard deviations calculated for the attributed, used in the normalization. Must have length equal to
        the number of attributes

    Returns
    -------
    weight_dict : dict
        A dicitonary of the edge weights
    '''
    
    pos_nodes = np.array(pos_nodes)
    att_nodes = np.array(att_nodes)
    if normalize_pos or pos_means is not None:
        pos_nodes = normalize_values(pos_nodes, pos_means, pos_stds)
    if normalize_att or att_means is not None:
        att_nodes = normalize_values(att_nodes, att_means, att_stds)
    if att_idx is None:
        # Use all attributes
        att_idx = ...
    
    weight_dict = {}
    for edge in nxgraph.edges:
        node1 = edge[0]
        node2 = edge[1]
        pos_node1 = pos_nodes[node1]
        pos_node2 = pos_nodes[node2]
        att_node1 = att_nodes[node1, att_idx]
        att_node2 = att_nodes[node2, att_idx]
        
        weight = calculate_weight(pos_node1, pos_node2, att_node1, att_node2, alpha)
        weight_dict[edge] = weight

    return weight_dict   