"""
The cross_validation module. This module implements nested cross validation.
The outer cross validation algorithm is used to give a less biased, lower variance
estimate of the performance metrics being used to evaluate our model. The nested
cross validation algorithm is used to tune BERT hyperparameters in a way that doesn't
result in overfitting the validation set.
"""
import pickle
import random
import numpy as np
from sklearn.model_selection import GroupKFold
from preprocessing.create_datasets import CookieTheftDataSet
from experiment.fine_tuning_bert import train_bert 
from experiment.evaluate import evaluate_scores
from torch.utils.data import DataLoader
DBANK_PICKLE_PATH = '/home/thiago/BERTFineTuning/preprocessing/data/pickled_data/dbank.pkl'

def cross_validation(k, num_hyp, hyp_epochs, patience):
    """
    The outer cross_validation algorithm. Logs training info to saves/training_log,
    and saves test scores on each fold to saves/cross_validation_scores. Optimal hyperparameter
    values can be found by looking at the names of the result files in saves/training_log.

    Parameters
    ----------
    k: int
        Number of folds to use
    num_hyp: int
        Number of times to randomly sample hyperparameters
    hyp_epochs: int
        Number of epochs to train model on randomly sampled hyperparameters
    patience: int
        Number of epochs without a significant increase in performance needed before
        terminating early stopping procedure
    """
    
    with open(DBANK_PICKLE_PATH, 'rb') as fp:
        dbank = pickle.load(fp)

    pids = dbank['pid']
    X = np.array(dbank['text'])
    y = np.array(dbank['label'])

    group_kfold = GroupKFold(n_splits=5).split(X, y, groups=pids) 
    data = []

    for _, fold_index in group_kfold:
        fold = {}
        fold["X"] = X[fold_index]
        fold["y"] = y[fold_index]
        data.append(fold)
    
    scores = {'Accuracy':0, 'F1':0, 'Precision':0, 'Recall':0, 'AUC':0}
    for test_idx in range(k):
        train_idx = [x for i, x in enumerate(range(k)) if i!=test_idx]
        train_folds = [data[x] for x in train_idx]
        test_fold = data[test_idx]

        # Find the optimal hyperparameters
        optimal_hyp = nested_cross_validation(train_folds, num_hyp, hyp_epochs, test_idx)
        # Find the optimal number of epochs to train our model
        optimal_epochs = early_stopping(train_folds, optimal_hyp, patience, test_idx)
        # Train our model using the training data 
        train_model(train_folds, optimal_hyp, optimal_epochs, test_idx)

        X_test = test_fold['X'].tolist()
        y_test = test_fold['y'].tolist()
        # Evaluate on test data
        model_scores = evaluate_scores(X_test, y_test,
        '/home/thiago/BERTFineTuning/saves/models/model_test_fold_{}'.format(test_idx))

        with open('/home/thiago/BERTFineTuning/saves/cross_validation_scores/scores.txt', 'a') as file:
            file.write('test key {}: '.format(test_idx) + str(model_scores) + '\n')
        
        scores['Accuracy'] += model_scores['Accuracy']
        scores['F1'] += model_scores['F1']
        scores['Precision'] += model_scores['Precision']
        scores['Recall'] += model_scores['Recall']
        scores['AUC'] += model_scores['AUC']

    for key in scores:
        scores[key] = scores[key]/k

    with open('/home/thiago/BERTFineTuning/saves/cross_validation_scores/scores.txt', 'a') as file:
            file.write('averaged scores: ' + str(scores) + '\n')


def train_model(folds, optimal_hyp, optimal_epochs, test_fold_num):
    """
    Trains a BERT model on all folds except the test fold for an optimal number of epochs
    using the optimal hyperparameters.

    Parameters
    ----------
    folds: list of dict
        A list of dictionaries containing train and test data
    optimal_hyp: dict
        The hyperparameters to use when fine-tuning BERT
    optimal_epochs: int
        The number of epochs to train the model for
    test_fold_num: int
        The index of the test fold for the outer cross validation algorithm (used for naming the results file)
    """
    X_train = []
    y_train = []
    for fold in folds:
        X_train = X_train + fold['X'].tolist()
        y_train = y_train + fold['y'].tolist()

    train_dataset = CookieTheftDataSet(X_train, y_train)
    train_dataloader = DataLoader (
        train_dataset, shuffle=True, batch_size=8
    )
    attn_dropout = optimal_hyp['attn_dropout'] 
    hidden_dropout = optimal_hyp['hidden_dropout'] 
    weight_init = optimal_hyp['weight_init'] 
    data_order_seed = optimal_hyp['data_order'] 
    lr = optimal_hyp['lr'] 
    
    _, _ = train_bert(
        train_corpus=X_train,
        train_labels=y_train,
        val_corpus=X_train[:1],
        val_labels=y_train[:1],
        dataloader=train_dataloader,
        lr=lr,
        num_epochs=optimal_epochs,
        patience =1e5,
        weight_init_seed=weight_init,
        data_order_seed=data_order_seed,
        hidden_dropout_prob=hidden_dropout,
        attention_dropout_prob=attn_dropout,
        save_model_at_end=True,
        save_models_path='/home/thiago/BERTFineTuning/saves/models/model_test_fold_{}'.format(test_fold_num),
        results_file_name='test_{}_training_AD_{}_HD_{}_WI_{}_DO_{}.txt'.format(test_fold_num,
                                                                                attn_dropout, 
                                                                                hidden_dropout, 
                                                                                weight_init,
                                                                                data_order_seed)) 


def early_stopping(folds, optimal_hyp, patience, test_fold_num):
    """
    Returns the optimal number of epochs to train the BERT model via 
    the early stopping algorithm. First randomly chooses a validation set. Then trains the BERT
    model on the remaining non-test-non-validation folds. Returns the epoch number at which
    training is halted by the early stopping mechanism.

    Parameters
    ----------
    folds: list of dict
        A list of dictionaries containing train and test data
    optimal_hyp: dict
        The hyperparameters to use when fine-tuning BERT
    patience: int
        Number of epochs without a significant increase in performance needed before
        terminating early stopping procedure
    test_fold_num: int
        The index of the test fold for the outer cross validation algorithm (used for naming the results file)

    Returns
    -------
    int
        The optimal number of epochs for training
    """
    val_idx = random.randint(0, len(folds) - 1)
    X_train, y_train, X_val, y_val, train_dataloader = create_dataloader(val_idx, folds)
    
    attn_dropout = optimal_hyp['attn_dropout']
    hidden_dropout = optimal_hyp['hidden_dropout']
    weight_init = optimal_hyp['weight_init']
    data_order_seed = optimal_hyp['data_order']
    lr = optimal_hyp['lr']

    _, num_epochs = train_bert(
        train_corpus=X_train,
        train_labels=y_train,
        val_corpus=X_val,
        val_labels=y_val,
        dataloader=train_dataloader,
        lr=lr,
        num_epochs=1000,
        patience = patience,
        weight_init_seed=weight_init,
        data_order_seed=data_order_seed,
        hidden_dropout_prob=hidden_dropout,
        attention_dropout_prob=attn_dropout,
        results_file_name='test_{}_early_stopping_AD_{}_HD_{}_WI_{}_DO_{}.txt'.format(test_fold_num,
                                                                                attn_dropout, 
                                                                                hidden_dropout, 
                                                                                weight_init,
                                                                                data_order_seed)) 
    return num_epochs
        
def nested_cross_validation(folds, num_hyp, hyp_epochs, test_fold_num):
    """
    A nested cross validation algorithm used for tuning hyperparameters.

    Parameters
    ----------
    folds: list of dict
        A list of dictionaries containing train and test data
    num_hyp: int 
        Number of times to randomly sample hyperparameters
    hyp_epochs: int
        Number of epochs to train model on randomly sampled hyperparameters
    test_fold_num: int
        The index of the test fold for the outer cross validation algorithm (used for naming the results file)

    Returns
    -------
    dict
        A dictionary containing the optimal hyperparameter values
    """

    best_hyperparams = {}
    best_cumulative_score = 0

    for h in range(num_hyp):
        # randomly sample hyperparameters.
        attn_dropout = random.randint(10, 40)/100
        hidden_dropout = random.randint(10, 40)/100
        weight_init = random.randint(0, 100)
        data_order_seed = random.randint(0, 100)
        lr = random.randint(1, 10) * (10**random.randint(-7, -5))

        cumulative_score = 0
        k = len(folds)
        for val_idx in range(k):
            X_train, y_train, X_val, y_val, train_dataloader = create_dataloader(val_idx, folds)
            top_val, _ = train_bert(
                train_corpus=X_train,
                train_labels=y_train,
                val_corpus=X_val,
                val_labels=y_val,
                dataloader=train_dataloader,
                lr=lr,
                num_epochs=hyp_epochs,
                patience = 1e5,
                weight_init_seed=weight_init,
                data_order_seed=data_order_seed,
                hidden_dropout_prob=hidden_dropout,
                attention_dropout_prob=attn_dropout,
                results_file_name='test_{}_nested_cross_val_AD_{}_HD_{}_WI_{}_DO_{}.txt'.format(test_fold_num,
                                                                                        attn_dropout, 
                                                                                        hidden_dropout, 
                                                                                        weight_init,
                                                                                        data_order_seed)) 
            cumulative_score += top_val

        if cumulative_score > best_cumulative_score:
            best_cumulative_score = cumulative_score
            best_hyperparams = {'attn_dropout': attn_dropout,
                                'hidden_dropout': hidden_dropout,
                                'weight_init': weight_init,
                                'data_order': data_order_seed,
                                'lr':lr}
    return best_hyperparams

def create_dataloader(val_idx, folds):
    """
    Returns the dataloader, along with the training and test data.

    Parameter
    ---------
    val_idx: int
        The index of the validation set
    folds: list of dict
        A list of dictionaries containing train and test data

    Returns
    -------
    list of strings
    list of int
    list of strings
    list of int
    torch.util.data.Dataloader
    """
    train_idx = [x for i, x in enumerate(range(len(folds))) if i!=val_idx]

    train_folds = [folds[x] for x in train_idx]
    val_fold = folds[val_idx]

    X_train = []
    y_train = []

    for fold in train_folds:
        X_train = X_train + fold['X'].tolist()
        y_train = y_train + fold['y'].tolist()
    X_val = val_fold['X'].tolist()
    y_val = val_fold['y'].tolist()

    train_dataset = CookieTheftDataSet(X_train, y_train)
    train_dataloader = DataLoader (
        train_dataset, shuffle=True, batch_size=8
    )

    return X_train, y_train, X_val, y_val, train_dataloader





            



        



    
    
    
        

