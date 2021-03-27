import numpy as np
from scipy.stats import zscore
from collections import deque
import matplotlib.cm as cm

def display_gland_numbers(G, break_mode = False, break_position = 10, print_number_of_glands=True):
    '''Display `G` (Graph) number of nodes demarcated or not and return lists of each class. 
    If `break_mode` is True it will break the function in the `break_position` number.'''
    
    count_break = 0

    nodes = []
    nodes_demarcated = []
    count_undefined = 0
       
    for node in G.nodes.items():
        node_idx = node[0]
        node_props = node[1]
        
        if node_props['demarcated'] == 'False':
            nodes.append(node)
        elif node_props['demarcated'] == 'True':
            nodes_demarcated.append(node)
        else:
            count_undefined += 1
            print('WARNING WRONG DATA!!!')

        if break_mode:
            count+=1
            if count == break_position:
                print(node_idx)
                print(node_props)
                break
    
    if print_number_of_glands:
        if len(nodes) > 0:
            print('Number of NORMAL glands: ', len(nodes))
        if len(nodes_demarcated) > 0:
            print('Number of DEMARCATED glands: ', len(nodes_demarcated))
        if count_undefined > 0:
            print('Number of UNDEFINED glands: ', count_undefined)
        
    return nodes, nodes_demarcated

  
def get_table_properties(G, properties, return_normalized=True):
    '''Return table `properties` of `nodes1` and `nodes2` unified with randomized `sample_quantity` normalized or not
    according to `return_normalized`. If `properties` contains 'idx' value it will not be normalized.'''
    
    table_properties = []
    classes = []
    for node in G.nodes(True):
        row_properties = []
        for prop in properties:
            row_properties.append(node[1][prop])
        
        classes.append(node[1]['demarcated']=='True')
        table_properties.append(row_properties)

    classes = np.array(classes)
    
    if not return_normalized:
        return np.array(table_properties), classes
    
    else:
        # Normalize values
        table_properties_normalized = np.zeros((len(table_properties), len(properties)))
        array_table_properties = np.array(table_properties)
        for col, prop in enumerate(properties):
            if prop == 'idx':
                table_properties_normalized[:,col] = array_table_properties[:,col]
            else:
                table_properties_normalized[:,col] = zscore(array_table_properties[:,col])

        return table_properties_normalized, classes   
    
def remove_idx_from_table_properties(properties_names, table_properties):
    if properties_names[0] == 'idx':
        table_properties = table_properties[:,1:]
    
    return table_properties

def get_color_inputs(G, measurment_prop='degree'):
    #https://colorbrewer2.org/?type=qualitative&scheme=Set1&n=5
    
    color_not_demarcated =  [55,126,184] #not demarcated    - classified as health
    color_demarcated     = [228,26,28]   #demarcated        - classified as unhealth
    color_undefined      = [77,175,74]   #undefined         - undefined value

    #inputs
    position = []
    measurements = []
    class_colors = []

    for node in G.nodes.items():
        node_idx = node[0]
        node_props = node[1]
        
        if node_props['demarcated'] == 'False':
            class_colors.append(color_not_demarcated)
        elif node_props['demarcated'] == 'True':
            class_colors.append(color_demarcated)
        else:
            class_colors.append(color_undefined)
    
        pos = [node_props['row'], node_props['column']]
        position.append(pos)
        
        measurements.append(node_props[measurment_prop])
    
    return position, class_colors, measurements

def color_objects(img_bin, positions, colors=None, values=None, colormap='viridis', print_progress=False):
    '''Color objects in binary image `img_bin`. `positions` must contain one pixel position
    inside each object.'''

    img_pad = np.pad(img_bin, 1, 'constant') 
    
    if (colors is None) and (values is None):
        raise ValueError("Either `colors` or `values` must be set")
    if colors is None:   
        if isinstance(colormap, str):
            cmap = cm.get_cmap(colormap)          
        else:
            try:
                colormap(0)
            except Exception:
                raise TypeError("Colormap is not callable")
            cmap = colormap

        values = np.array(values).astype(float)
        values = (values-values.min())/(values.max()-values.min())
        colors = cmap(values)[:,:-1]
        colors = np.round(255*colors).astype(int)
    
    positions = np.array(positions)
    positions += 1
    img_colored = np.zeros((img_bin.shape[0], img_bin.shape[1], 3), dtype=np.uint8)
    
    # Color each connected component using `positions` as seed
    for idx, pos in enumerate(positions):
        pixels = flood_fill(img_pad, pos, print_progress)
        pixels -= 1
        #img_colored[pixels[:,0], pixels[:,1]] = colors[idx]
        try:
            img_colored[pixels[:,0][pixels[:,0]<img_colored.shape[0]], pixels[:,1][pixels[:,1]<img_colored.shape[1]]] = colors[idx]
        except:
            print('except ERRO')

    return img_colored        

def flood_fill(img_bin, initial_pixel, print_progress=False):
    '''Find pixels inside connected componente using `initial_pixel` as seed.'''
    count = 0
    
    initial_pixel = tuple(initial_pixel)
    pixels_to_analyze = deque([initial_pixel])
    visited_pixels = set([initial_pixel])
    
    img_colored = np.zeros((img_bin.shape[0], img_bin.shape[1], 3), dtype=np.uint8)
    
    while len(pixels_to_analyze)>0: 
        
        current_pixel = pixels_to_analyze.popleft()
        
        if print_progress:
            if count % 128000 == 0:
                print(current_pixel)
                
        count+=1

        for neighbor in iterate_neighbors(current_pixel):
            if (img_bin[neighbor[0], neighbor[1]]>0) and (neighbor not in visited_pixels):
                pixels_to_analyze.append(neighbor)
                visited_pixels.add(neighbor)
    
    return np.array(list(visited_pixels))
    
def iterate_neighbors(pixel_coords):
    '''Iterate over neighbors of a given pixel'''
    
    neis = [(pixel_coords[0]-1, pixel_coords[1]-1), (pixel_coords[0]-1, pixel_coords[1]), 
            (pixel_coords[0]-1, pixel_coords[1]+1), (pixel_coords[0], pixel_coords[1]-1), 
            (pixel_coords[0], pixel_coords[1]+1), (pixel_coords[0]+1, pixel_coords[1]-1),
            (pixel_coords[0]+1, pixel_coords[1]), (pixel_coords[0]+1, pixel_coords[1]+1)]
    
    for n in neis:
        yield n    
        
def get_color_inputs_by_cross_results(G, classes_pred, test_index, graph_idx):
    #POSITIVE = demarcated as CA 3+3
    #NEGATIVE = not demarcated in CA region   
    blue_color        = [ 44,123,182]  #green  #not demarcated    - classified as not demarcated - TRUE  NEGATIVE
    light_blue_color  = [171,217,233]  #blue   #not demarcated    - classified as     demarcated - FALSE NEGATIVE
    red_color         = [255, 0, 0]    #demarcated        - classified as     demarcated - TRUE  POSITIVE
    light_red_color   = [255, 255, 0]  #yellow #demarcated        - classified as not demarcated - FALSE POSITIVE
    
    #inputs
    position = []
    measurements = []
    class_colors = []

    #confusion matrix
    confusion_matrix = []
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0
    
    list_nodes = G.nodes()

    for idx_array, idx_test in enumerate(test_index):
        g_idx = str(int(graph_idx[idx_test]))
        node_props = list_nodes[g_idx]

        #data validation
        #print('{} - {} - {}'.format(idx_array, node_props['demarcated'], classes_pred[idx_array]))

        if node_props['demarcated'] == 'False':
            if classes_pred[idx_array] == 1:
                class_colors.append(light_blue_color)
                false_negative += 1
            elif classes_pred[idx_array] == 0:
                class_colors.append(blue_color)
                true_negative += 1
        elif node_props['demarcated'] == 'True':
            if classes_pred[idx_array] == 1:
                class_colors.append(red_color)
                true_positive += 1
            elif classes_pred[idx_array] == 0:
                class_colors.append(light_red_color)
                false_positive += 1

        pos = [node_props['row'], node_props['column']]
        position.append(pos)

        measurements.append(node_props['area'])
        
    confusion_matrix = [ [true_negative, false_negative],
                         [false_positive, true_positive] ]

    return position, class_colors, measurements, confusion_matrix


def get_colors_by_pred_results(G, classes_pred):

    '''Get gland colors according to prediction classes.
    
    Parameters
    ----------
    G : NetworkX Graph
        Graph with nodes containing glands properties.
    classes_pred : numpy array
        Array containing the prediciton classes with the same size of nodes in graph.
    Returns
    -------
    position : numpy array
        Center of mass position to colorize gland.
    class_colors : numpy array
        Gland color according to prediction value.
    confusion_matrix
        Confusion matrix with color results.
        [ [ true_negative, false_positive],
          [false_negative,  true_positive] ]
    '''
    
    #outputs
    position = []
    class_colors = []
    confusion_matrix = []


    #NEGATIVE = not demarcated in CA region   
    #POSITIVE = demarcated as CA 3+3
    blue_color       = [ 44,123,182]    #green #not demarcated    - classified as not demarcated - TRUE  NEGATIVE
    light_blue_color = [171,217,233]    #not demarcated    - classified as     demarcated - FALSE NEGATIVE
    red_color        = [215, 25, 28]    #demarcated        - classified as     demarcated - TRUE  POSITIVE
    light_red_color  = [253,174, 97]    #yellow #demarcated        - classified as not demarcated - FALSE POSITIVE


    true_negative  = 0
    false_negative = 0
    true_positive  = 0
    false_positive = 0

    list_nodes = G.nodes()

    for g_idx, pred in enumerate(classes_pred):
        node_props = list_nodes[str(g_idx)]

        if node_props['demarcated'] == 'False':
            if classes_pred[g_idx] == 0:
                class_colors.append(blue_color)
                true_negative += 1
            elif classes_pred[g_idx] == 1:
                class_colors.append(light_blue_color)
                false_negative += 1
        elif node_props['demarcated'] == 'True':
            if classes_pred[g_idx] == 1:
                class_colors.append(red_color)
                true_positive += 1
            elif classes_pred[g_idx] == 0:
                class_colors.append(light_red_color)
                false_positive += 1

        pos = [node_props['row'], node_props['column']]
        position.append(pos)

    confusion_matrix = [ [ true_negative, false_positive],
                         [false_negative,  true_positive] ]

    return position, class_colors, confusion_matrix

def get_colors_by_pred_results_test_indices(G, classes_pred, test_indices):

    '''Get gland colors according to prediction classes.
    
    Parameters
    ----------
    G : NetworkX Graph
        Graph with nodes containing glands properties.
    classes_pred : numpy array
        Array containing the prediciton classes with the same size of nodes in graph.
    Returns
    -------
    position : numpy array
        Center of mass position to colorize gland.
    class_colors : numpy array
        Gland color according to prediction value.
    confusion_matrix
        Confusion matrix with color results.
        [ [ true_negative, false_positive],
          [false_negative,  true_positive] ]
    '''
    
    #outputs
    position = []
    class_colors = []
    confusion_matrix = []


    #NEGATIVE = not demarcated in CA region   
    #POSITIVE = demarcated as CA 3+3
    #COLOR SCHEME - https://colorbrewer2.org/?type=qualitative&scheme=Set1&n=5
    color_not_demarcated_t = [55,126,184]   #not demarcated    - TRUE NEGATIVE  (HEALTHY)
    color_not_demarcated_f = [255,240,30]   #not demarcated    - FALSE NEGATIVE (WRONG HEALTHY)
    color_demarcated_t     = [228,26,28]    #demarcated        - TRUE  POSITIVE (UNHEALTHY)
    color_demarcated_f     = [57,184,55]    #demarcated        - FALSE POSITIVE (WRONG UNHEALTHY)


    true_negative  = 0
    false_negative = 0
    true_positive  = 0
    false_positive = 0

    list_nodes = G.nodes()

    for idx, g_idx in enumerate(test_indices):
        node_props = list_nodes[str(g_idx)]

        if node_props['demarcated'] == 'False':
            if classes_pred[g_idx] == 0:
                class_colors.append(color_not_demarcated_t)
                true_negative += 1
            elif classes_pred[g_idx] == 1:
                class_colors.append(color_not_demarcated_f)
                false_positive += 1
        elif node_props['demarcated'] == 'True':
            if classes_pred[g_idx] == 1:
                class_colors.append(color_demarcated_t)
                true_positive += 1
            elif classes_pred[g_idx] == 0:
                class_colors.append(color_demarcated_f)
                false_negative += 1

        pos = [node_props['row'], node_props['column']]
        position.append(pos)

    confusion_matrix = [ [ true_negative, false_positive],
                         [false_negative,  true_positive] ]

    return position, class_colors, confusion_matrix