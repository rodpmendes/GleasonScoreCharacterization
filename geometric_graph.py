import numpy as np
from scipy.stats import zscore
from scipy import ndimage as ndi
from igraph import Graph
from scipy.spatial import cKDTree as kdtree

def geometric_graph(positions, radius):

    tree = kdtree(positions)
    edges = list(tree.query_pairs(radius))
    g = Graph(n=len(positions), edges=edges)

    return g

def network_from_mask(img_mask, radius):
    '''Calculate voronoi network from a given mask image.'''
    
    lbl, nro = ndi.label(img_mask)
    idx = np.array(range(1, nro+1, 1))
    cm = ndi.measurements.center_of_mass(img_mask, lbl, idx)
    
    #ROTATE CENTER OF MASS
    cm = np.array(cm)
    cm = cm[:,::-1]
    cm[:,1] = img_mask.shape[0]-cm[:,1]

    g = geometric_graph(cm, radius)
    
    return g, cm

if __name__=="__main__":

    # Test the code with random points
    import misc
    import networkx as nx
    import networkx.drawing as draw

    p = np.random.rand(300, 2)
    g = geometric_graph(p, 0.08)

    gnx = misc.igraph_to_nx(g)
    draw.draw_networkx(gnx, p, node_size=8, with_labels=False)