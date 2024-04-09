import os
import gc
import time

import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers

from utils import load_data, pickle_load, format_filename, write_log, prepare_smiles, prepare_labels
from models import Our
from config import ModelConfig, PROCESSED_DATA_DIR,  ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    DRUG_VOCAB_TEMPLATE

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(train_d,dev_d,test_d,kfold,dataset, neighbor_sample_size, embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch, embed_smile,inter_types, callbacks_to_add=None, overwrite=True):
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset=dataset
    config.K_Fold=kfold
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add
    config.len_train = len(train_d)
    config.len_smile = len(embed_smile[0])
    config.inter_types = inter_types
    config.drug_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=dataset)))
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset)))
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset)))
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset))
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset))
    config.exp_name = f'our_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str
    
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = Our(config)

    train_data=np.array(train_d)
    valid_data=np.array(dev_d)
    test_data=np.array(test_d)

    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()

        x_train_drug_one = train_data[:, :1]
        x_train_drug_two = train_data[:, 1:2]
        x_train_drug_one_smile = prepare_smiles(x_train_drug_one, embed_smile)
        x_train_drug_two_smile = prepare_smiles(x_train_drug_two, embed_smile)

        x_valid_drug_one = valid_data[:, :1]
        x_valid_drug_two = valid_data[:, 1:2]
        x_valid_drug_one_smile = prepare_smiles(x_valid_drug_one, embed_smile)
        x_valid_drug_two_smile = prepare_smiles(x_valid_drug_two, embed_smile)

        y_train=prepare_labels(train_data[:, 2:3],inter_types)
        y_valid=prepare_labels(valid_data[:, 2:3],inter_types)

        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2],x_train_drug_one_smile,x_train_drug_two_smile], 
                  y_train=y_train,

                  x_valid=[valid_data[:, :1], valid_data[:, 1:2],x_valid_drug_one_smile,x_valid_drug_two_smile], 
                  y_valid=y_valid)

        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    x_one_smile = prepare_smiles(valid_data[:, :1], embed_smile)
    x_two_smiles = prepare_smiles(valid_data[:, 1:2], embed_smile)
    x_one = [valid_data[:, :1], valid_data[:, 1:2], x_one_smile, x_two_smiles]
    
    y_two = prepare_labels(valid_data[:, 2:3], inter_types)

    
    auc, acc, f1, aupr, pre, rec = model.score(x=x_one, y=y_two)

    print(f'Logging Info - dev_auc: {auc}, dev_acc: {acc}, dev_f1: {f1}, dev_aupr: {aupr}, dev_rec: {rec}, dev_pre: {pre} ')
    train_log['dev_auc'] = auc
    train_log['dev_acc'] = acc
    train_log['dev_f1'] = f1
    train_log['dev_aupr']=aupr
    train_log['k_fold']=kfold
    train_log['dataset']=dataset
    train_log['aggregate_type']=config.aggregator_type
    train_log['dev_rec'] = rec
    train_log['dev_pre'] = pre

    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        auc, acc, f1,aupr = model.score(x=[valid_data[:, :1], valid_data[:, 1:2], 
                                        prepare_smiles(valid_data[:, :1], embed_smile), prepare_smiles(valid_data[:, 1:2], embed_smile)], 
                                        y=valid_data[:, 2:3])

        train_log['swa_dev_auc'] = auc
        train_log['swa_dev_acc'] = acc
        train_log['swa_dev_f1'] = f1
        train_log['swa_dev_aupr']=aupr
        print(f'Logging Info - swa_dev_auc: {auc}, swa_dev_acc: {acc}, swa_dev_f1: {f1}, swa_dev_aupr: {aupr}') 
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()

    y = prepare_labels(test_data[:, 2:3], inter_types)

    auc, acc, f1, aupr, pre, rec  = model.score(x=[test_data[:, :1], test_data[:, 1:2], 
                                        prepare_smiles(test_data[:, :1], embed_smile), prepare_smiles(test_data[:, 1:2], embed_smile)], 
                                        y=y)
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_aupr'] =aupr
    train_log['test_rec'] = rec
    train_log['test_pre'] = pre

    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}, dev_rec: {rec}, dev_pre: {pre} ')
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, f1,aupr = model.score(x=[test_data[:, :1], test_data[:, 1:2], prepare_smiles(test_data[:, :1], embed_smile), prepare_smiles(test_data[:, 1:2], embed_smile)], 
                                        y=test_data[:, 2:3])
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        print(f'Logging Info - swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_f1: {f1}, swa_test_aupr: {aupr}')
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

