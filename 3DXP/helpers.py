import numpy as np
import os
import errno
import io
import scipy.sparse
import scipy.io
import json
import csv
import pandas as pd

MEASURES_AT_K = ['precision@k', 'recall@k', 'f1@k']


def binarize(scores, train, k=5):
    '''
    Transform scores matrix to predictions matrix,
    by returning for each user the top-K items 
    not in the training matrix
    
    Parameters:
    
    '''
    # Set items in training to -infinity
    scores_without_train = np.asarray(scores)
    scores_without_train[train.toarray().astype(bool)] = -1.e8  # -infinity
    
    # For each user (row) return indices of top-K items
    best_k_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    
    # Create scipy sparse matrix
    prediction_matrix = scipy.sparse.csr_matrix((scores.shape[0], scores.shape[1]))
    for i, indices in enumerate(best_k_idx):
        prediction_matrix[i, indices] = 1
        
    return prediction_matrix
    
    

def save_csr(fname, csr, format_csv=True):
    '''
    Saves the sparse matrix csr into the file fname.
    fname -- string
        name of the file where to save the sparse matrix
    csr -- sparse matrix scipy.sparse.csr.csr_matrix
        the sparse matrix to save
    format_csv -- bool
        if True, save to csv format, else save to .mtx format
    '''
    if format_csv:
        # Densify to np.array to convenience:
        dense_csr = csr.toarray()
        df = pd.DataFrame({'ID': np.arange(dense_csr.shape[0]) + 1})
        for k in range(dense_csr.shape[1]):
            df[str(k + 1)] = dense_csr[:, k]
        df.to_csv(path_or_buf=fname + '.csv', sep=';', index=False)
    else:
        return scipy.io.mmwrite(fname, csr)


def load_csr(fname):
    '''
    Load the CSR matrix from the file given as argument, which can be either 
    the name of .mtx file or the name of .csv file
    '''
    name_inspect = fname.split('.')
    if len(name_inspect)==2 and name_inspect[1]=='csv':
        df = pd.read_csv(fname, sep=';')
        X = df.as_matrix()[:, 1:]  # remove the column 'ID'
        return scipy.sparse.csr_matrix(X)
    else:
        return scipy.sparse.csr_matrix(scipy.io.mmread(fname))
    
    
def load_splits(path):
    print 'Loading splits from', path
    splits = []    
    i = 0
    while True:
        try:
            print 'Reading split', i
            train_csr = load_csr("{}/{}-train".format(path, i))
            test_csr = load_csr("{}/{}-test".format(path, i))
            splits.append((train_csr, test_csr))
            i += 1
        except IOError:
            print 'Found {} splits in total'.format(i)
            break
    try:
        with open('{}/meta.json'.format(path), 'r') as fp:
            meta = json.load(fp)
    except Exception:
        meta = None
        print 'No metadata found'
    return splits, meta   


def scores_to_sorted_indices(scores):
    '''
    Return indices of sorted scores (descending)

    Example:
        if scores = [3, 1, 2, 4, 0]
        then return ranked_indices = [3, 0, 2, 1, 4]    
    '''
    zipped = zip(scores, list(range(len(scores))))
    sorted_zipped = sorted(zipped, reverse=True)
    sorted_vals, sorted_indices = zip(*sorted_zipped)
    return np.asarray(sorted_indices)    

	
def zip_map(rows, metric, *args, **kwargs):
    '''
    Unpack each element of rows, apply function metric and store result in list.
    
        metric(*(row+args), **kwargs)
    
    args and kwargs will be broadcast
    '''
    scores = []    
    for row in rows:
        score = metric(*(row+args), **kwargs)
        scores.append(score)
    return scores
	

def measures_at_k_from_indices(stripped_sorted_indices, unary_ratings, k):
    '''
    Compute measures@k for unary ratings for one user.
    
    stripped_sorted_indices: list of indices of k-best predicted items
        must be numpy.arrays
    unary_ratings: list of ground truth unary ratings (True/False) for each item
    k: number of items to consider
    
    Return
        precision@k, recall@k, f1
    '''
    precision = unary_ratings[stripped_sorted_indices].mean()
    recall = unary_ratings[stripped_sorted_indices].sum() / float(unary_ratings.sum())
    if precision + recall > 0.:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
    return np.asarray([precision, recall, f1])
    
    
def measures_at_k(pred_scores, unary_ratings, k, fast_mode=True):
    '''
    Compute measures@k for unary ratings for one user.
    
    pred_score: list of predicted scores for each item
        must be numpy.array
    unary_ratings: list of ground truth unary ratings (True/False) for each item
    k: number of items to consider
    
    Return
        precision@k, recall@k, f1
    '''
    if fast_mode:
        # get unsorted indices of k largest
        stripped_sorted_indices = np.argpartition(pred_scores, -k)[-k:]
    else:
        sorted_indices = evalrec.scores_to_sorted_indices(pred_scores)
        stripped_sorted_indices = sorted_indices[:k]
    measures = measures_at_k_from_indices(stripped_sorted_indices, unary_ratings, k)
    return measures


def print_metrics(all_predictions, train, test, k=5, n_eval=None, verbose=True):
    
    def vprint(arg):
        if verbose:
            print arg
    
    vprint('Predictions shape {}'.format(all_predictions.shape))
        
    all_test_predictions = np.array(all_predictions)
    all_test_predictions[train.toarray().astype(bool)] = -1.e8  # -infinity
    
    
    vprint('K={}'.format(k))
    vprint('Metrics {}'.format(MEASURES_AT_K))
    train_scores = batch_measures_at_k(all_predictions, train, k, n_eval)
    vprint('Train {}'.format(np.mean(train_scores, axis=0)))
    test_scores = batch_measures_at_k(all_test_predictions, test, k, n_eval)
    vprint('Test {}'.format(np.mean(test_scores, axis=0)))
    vprint('Shapes. Train {} Test {}'.format(train_scores.shape, test_scores.shape))
    
    return np.mean(train_scores, axis=0), np.mean(test_scores, axis=0)

def get_precision_at_k(all_test_predictions, test, k=5):
    all_test_predictions = all_test_predictions.toarray()  # need dense
    test_scores = np.mean(batch_measures_at_k(all_test_predictions, test, k, None), axis=0)
    return test_scores[0]
    
def batch_measures_at_k(all_predictions, all_unary_ratings, k, n_eval=None, fast_mode=True):
    '''
    Always use fast mode. Slow mode for debugging.
    '''
    if isinstance(all_unary_ratings, scipy.sparse.csr.csr_matrix):
        all_unary_ratings = all_unary_ratings.toarray()
    scores = []
    if fast_mode:
        best_k_idx = np.argpartition(all_predictions, -k, axis=1)[:, -k:]
    for i, (pred_scores, unary_ratings) in enumerate(zip(all_predictions, all_unary_ratings)):
        if n_eval and i >= n_eval:
            break
        if fast_mode:
            score = measures_at_k_from_indices(best_k_idx[i], unary_ratings, k)
        else:
            score = measures_at_k(pred_scores, unary_ratings, k)
        scores.append(score)
    return np.asarray(scores)
    
