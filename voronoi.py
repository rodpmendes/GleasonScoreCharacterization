import numpy as np
import Polygon
from scipy.spatial import Voronoi
from igraph import Graph
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import networkx as nx
import networkx.drawing as draw
from scipy import ndimage

def voronoi_network(points=None, N=None, allowedRegion=None):
    """Generate Voronoi network
    
    Parameters
    ----------
    points : array_like
      Position of the points. Each row contains the (x,y) position of a point.
    N : int
      Number of points (ignored if points is not None).
    allowedRegion : array_like
      Square bounds of the Voronoi tessellation.
    """

    if N is None and points is None:
        raise ValueError("Either N or points need to be specified")
    elif points is None:
        points = np.random.rand(N, 2)
    else:
        N = len(points)

    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)

    if allowedRegion is None:
        allowedRegion = np.array([(xmin, ymin), (xmax, ymin), 
                                  (xmax, ymax), (xmin, ymax)])


    regionPol = Polygon.Polygon(allowedRegion)

    scale = np.maximum(xmax-xmin, ymax-ymin)
    additionalPoints = [(xmin-scale, ymin-scale),(xmax+scale, ymin-scale),
                        (xmax+scale, ymax+scale),(xmin-scale, ymax+scale)]

    temporaryPoints = np.array(points.tolist() + additionalPoints)

    vor = Voronoi(temporaryPoints)

    cellCollection = []
    isBorder = []
    regions = vor.regions
    for point in vor.point_region:
        if (len(regions[point])>0):
            if (np.min(regions[point])!=-1):

                cell = vor.vertices[regions[point]]
                cellPoly = Polygon.Polygon(cell)
                resPoly = (cellPoly&regionPol)
                resCell = np.array(resPoly[0])

                cellCollection.append(resCell)

                if (cellPoly-resPoly).area()>0:
                    isBorder.append(1)
                else:
                    isBorder.append(0)


    edges = vor.ridge_points

    # Remove edges of neighbooring cells outside desired region
    edges2remove = []
    for edgeIndex, vertices in enumerate(vor.ridge_vertices):
        shouldRemove = True
        if vertices[0]>=0:
            v = vor.vertices[vertices[0]]
            if regionPol.isInside(v[0], v[1])==True:
                shouldRemove = False
        if vertices[1]>=0:
            v = vor.vertices[vertices[1]]
            if regionPol.isInside(v[0], v[1])==True:
                shouldRemove = False

        if shouldRemove==True:
            edges2remove.append(edgeIndex)

    g = Graph(edges=edges.tolist())
    g.delete_edges(edges2remove)
    tempPointsIndices = range(len(points), len(points)+4)
    g.delete_vertices(tempPointsIndices)

    g.vs['isBorder'] = isBorder
    g.vs['pos'] = points.tolist()

    return g, cellCollection

def plot_voronoi(cellCollection, ax):
    """Plot Voronoi network

    Parameters
    ----------
    cellCollection : list
      List containing a set of polygons returned by the function voronoi_network.
    ax : matplotlib axes
      axes to plot the network.
    """

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    coll = PolyCollection(cellCollection, closed=True, facecolors='none', edgecolors='b', linestyle='--', alpha=0.2)
    ax.add_collection(coll)
    ax.autoscale_view()

    return ax

def plot_data(points, g, cellCollection, ax):
    """Plot Voronoi network indicating which nodes are at the border of the tessellation

    Parameters
    ----------
    points : array_like
      Array containing the positions of the points
    g : igraph graph
      igraph Graph object containing the graph to plot
    cellCollection : list
      List containing a set of polygons returned by the function voronoi_network
    ax : matplotlib axes
      axes to plot the network
    """
    
    plt.scatter(points[:,0], points[:,1], c=g.vs['isBorder'], s=30, axes=ax, zorder=10)
    fig = plot_voronoi(cellCollection, ax)

    gnx = nx.Graph(g.get_edgelist())
    draw.draw_networkx_edges(gnx, points)

def voronoi_from_mask(img_mask):
    '''Calculate voronoi network from a given mask image.'''
    
    lbl, nro = ndimage.label(img_mask)
    idx = np.array(range(1, nro+1, 1))
    cm = ndimage.measurements.center_of_mass(img_mask, lbl, idx)
    
    #ROTATE CENTER OF MASS
    cm = np.array(cm)
    cm = cm[:,::-1]
    cm[:,1] = img_mask.shape[0]-cm[:,1]

    vor = Voronoi(cm)
    points = vor.points
    g, cell_collection = voronoi_network(points)
    
    return g, points, cell_collection