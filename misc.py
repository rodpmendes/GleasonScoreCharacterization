import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.drawing as draw
    
def PCA(X, new_dim, use_cov=False):
    """
    IN:
    X: [N,M] array, where N is the number of objects and M the number of features
    new_dim: Dimension of the projected data (number of PCA features)
    useCov: Whether to use covariance matrix (True) or correlation matrix (False)

    OUT:
    PCA_features: [N,new_dim] array, new features
    eigenvalues: Eigenvalues in decreasing order
    main_eigenvectors: [N,new_dim] array, Eigenvectors in each column
    """
    
    # Use covariance matrix or correlation matrix
    if use_cov:
        C = np.cov(X.T)
    else:
        C = np.corrcoef(X.T)

    # Obtain the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Get indices that will be used to sort the eigenvalues in decreasing order
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Sort the eigenvalues and respective eigenvectors
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]

    # Keep only the L first eigenvectors
    main_eigenvectors = eigenvectors[:,0:new_dim]  

    # Subtract the mean of each feature
    u = np.mean(X,axis=0)
    X_norm = X-u

    # Also divide by the standard deviation
    if (use_cov==False):
        stdev = np.std(X, axis=0, ddof=1)
        X_norm = X_norm/stdev
    
    # Project the data into new variables (PCA features)
    PCA_features = np.dot(X_norm, main_eigenvectors)
    
    return PCA_features, eigenvalues, main_eigenvectors    

def mask_correction(mask):
    if (len(np.unique(mask)) > 2):
        mask[mask < 128] = 0
        mask[mask >= 128] = 255
        mask = 255-mask
    
    if(len(np.unique(mask)) != 2):
        print('ERROR, CHECK IMAGE MASK')
    
    return mask

def plot_graph(nxgraph, pos, weights, img_mask=None, min_width=1, max_width=10, title='', path_result=None, plt_figsize=(24,16), plt_node_size=10, alpha=.6, show_edges=True):

    pos = np.array(pos)
    weights = np.array(weights)

    min_weight = np.min(weights)
    max_weight = np.max(weights)

    weights_norm = (weights - min_weight) / (max_weight - min_weight)
    width = (weights_norm * (max_width - min_width)) + min_width

    plt.figure(figsize=plt_figsize)
    ax = plt.subplot(111, aspect='equal', title=title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('off')

    pos_img = pos.copy()
    if img_mask is not None:
        pos_img[:,1] = img_mask.shape[0]-pos_img[:,1]
        plt.imshow(img_mask, interpolation='nearest', cmap='gray')

    if show_edges:
        arcsGD = nx.draw_networkx_edges(nxgraph, pos_img, width = width, ax = ax, edge_color='b', alpha=alpha)
        
    arcsGD = nx.draw_networkx_nodes(nxgraph, pos_img, width = width, ax = ax, node_color='r', node_size=plt_node_size)

    if path_result is not None:
        plt.savefig(path_result, dpi=300)

def show_img(img, title='', cmap='gray'):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', title=title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.imshow(img, cmap=cmap)

def igraph_to_nx(g):
    
    '''Create NetworkX Graph'''

    nxgraph = nx.Graph()
    for i in range(g.vcount()):
        nxgraph.add_node(i)

    for edge in g.es:
        nxgraph.add_edge(edge.source, edge.target)
        
    return nxgraph

def split_dataset(properties, classes, n_train, n_validate):
    '''Split dataset into training and validation. Array 'properties'
       contain the measurements, each row is an object and each column
       corresponds to a property. 'classes' array with classes index (elements with index)
       in the data. 'n_train' is the number of objects used for training. 'n_validate' is the number 
       of objects used for validate'''

    n_classes = np.max(classes) + 1
    properties_train = []
    properties_validate = []
    classes_train = []
    classes_validate = []
    for class_index in range(n_classes):
        ind = np.nonzero(classes==class_index)[0]
        ind_train = np.random.choice(ind, size=n_train, replace=False)
        ind_validate = list(set(ind) - set(ind_train))

        properties_train_class = properties[ind_train]
        properties_validate_class = properties[ind_validate]

        properties_train.extend(properties_train_class.tolist())
        properties_validate.extend(properties_validate_class.tolist())

        classes_train.extend([class_index]*n_train)
        classes_validate.extend([class_index]*n_validate)
    
    return (np.array(properties_train), np.array(properties_validate), 
           np.array(classes_train), np.array(classes_validate))