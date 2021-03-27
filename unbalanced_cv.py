import numpy as np

# Only works for two classes

def get_fold_indices(fold_index, fold_size, num_folds, N):
    '''Given a fold index and the size, number of folds and number of objects in the dataset,
    returns the indices of the objects in the fold. These indices can be used to obtain the
    measurements of the objects in the data matrix.'''
    
    if fold_index<num_folds-1:
        return list(range(fold_index*fold_size, (fold_index+1)*fold_size))
    else:
        return list(range(fold_index*fold_size, N))
    
def get_splits(Ns, Nl, folds):
    '''Given the number of objects in the smaller class (Ns) and in the larger class (Nl) and the number of folds,
    returns the indices of all the folds used in the unbalanced cross-validation.'''

    fold_size = int(round(Ns/folds))
    if (Ns-fold_size*folds) > ((fold_size+1)*folds-Ns):
        fold_size += 1

    num_folds_larger = int(Nl/fold_size)
    if (Nl-fold_size*num_folds_larger) > (fold_size*(num_folds_larger+1)-Nl):
        num_folds_larger += 1

    num_folds_test = np.array([num_folds_larger//folds]*folds)
    if num_folds_larger%folds>0:
        num_folds_test[:num_folds_larger%folds] += 1
    cumsum_num_folds_test = [0] + np.cumsum(num_folds_test).tolist()

    #print(folds, fold_size, num_folds_larger)

    fold_inds_smaller = set(range(folds))
    fold_inds_larger = set(range(num_folds_larger))
    splits = []
    for fold_idx in range(folds):

        fold_inds_smaller_test = fold_idx
        fold_inds_smaller_train = fold_inds_smaller - set([fold_idx])

        fold_inds_larger_test = list(range(cumsum_num_folds_test[fold_idx],cumsum_num_folds_test[fold_idx+1]))
        fold_inds_larger_train_pool = list(fold_inds_larger - set(fold_inds_larger_test))
        fold_inds_larger_train = sorted(np.random.choice(fold_inds_larger_train_pool, folds-1, False))

        inds_smaller_test = get_fold_indices(fold_inds_smaller_test, fold_size, folds, Ns)
        inds_smaller_train = []
        for idx in fold_inds_smaller_train:
            inds_smaller_train.extend(get_fold_indices(idx, fold_size, folds, Ns))

        inds_larger_test = []
        for idx in fold_inds_larger_test:
            inds_larger_test.extend(get_fold_indices(idx, fold_size, num_folds_larger, Nl))

        inds_larger_train = []
        for idx in fold_inds_larger_train:
            inds_larger_train.extend(get_fold_indices(idx, fold_size, num_folds_larger, Nl))
            
        splits.append([inds_smaller_train, inds_larger_train, inds_smaller_test, inds_larger_test])

        '''print(fold_inds_smaller_train)
        print(inds_smaller_train)
        print(fold_inds_smaller_test)
        print(inds_smaller_test)
        print(fold_inds_larger_train)
        print(inds_larger_train)
        print(fold_inds_larger_test)
        print(inds_larger_test)
        print('\n')'''
        
    return splits

def get_fold(data, classes, num_folds):
    '''Yields the folds of an unbalanced cross-validation.
    
    Parameters
    ----------
    data : numpy array
        Data matrix where each column represents a feature and each row an object.
    classes : numpy array
        Array containing the classes for each row of `data`. Length should be the same as
        the number of rows in `data`.
    num_folds : int
        Number of folds for cross-validation
    Returns
    -------
        Generator containing for each call of the function:
    data_train : numpy array
        Data used for training the classifier. Contains the same number of objects from each class.
    classes_train : numpy array
        Classes of the training data
    data_test : numpy array
        Data used for testing the classifier. May contain different number of objects from each class.
    classes_test : numpy array
        Classes of the test data
    indices_test : list
        Indices of the test data in matrix `data`
    '''
    
    data = np.array(data)
    classes = np.array(classes)

    num_elem_in_class = np.bincount(classes)
    smaller_class = np.argmin(num_elem_in_class)
    larger_class = 1 - smaller_class

    inds_smaller = np.nonzero(classes==smaller_class)[0]
    inds_larger = np.nonzero(classes==1-smaller_class)[0]

    Ns = len(inds_smaller)
    Nl = len(inds_larger)

    splits = get_splits(Ns, Nl, num_folds)

    data_smaller = data[inds_smaller]
    data_larger = data[inds_larger]

    perm_smaller = np.random.permutation(Ns)
    perm_larger = np.random.permutation(Nl)
    data_smaller = data_smaller[perm_smaller]
    data_larger = data_larger[perm_larger]

    for split in splits:

        inds_smaller_train, inds_larger_train, inds_smaller_test, inds_larger_test = split
        data_train = np.concatenate((data_smaller[inds_smaller_train], data_larger[inds_larger_train]), axis=0)
        classes_train = np.concatenate(([smaller_class]*len(inds_smaller_train), 
                                        [larger_class]*len(inds_larger_train)))

        data_test = np.concatenate((data_smaller[inds_smaller_test], data_larger[inds_larger_test]), axis=0)
        classes_test = np.concatenate(([smaller_class]*len(inds_smaller_test), 
                                        [larger_class]*len(inds_larger_test)))

        or_ind_smaller = inds_smaller[perm_smaller[inds_smaller_test]].tolist()
        or_ind_larger = inds_larger[perm_larger[inds_larger_test]].tolist()
        indices_test = or_ind_smaller+or_ind_larger
        
        yield data_train, classes_train, data_test, classes_test, indices_test