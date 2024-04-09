import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'


KG_FILE = {
           'kegg':os.path.join(RAW_DATA_DIR,'kegg','train2id.txt'),
           'our':os.path.join(RAW_DATA_DIR,'our_multi','train2id.txt')}
ENTITY2ID_FILE = {
                    'kegg':os.path.join(RAW_DATA_DIR,'kegg','entity2id.txt'),
                    'our':os.path.join(RAW_DATA_DIR,'our_multi','entity2id.txt')}
EXAMPLE_FILE = {
               'kegg':os.path.join(RAW_DATA_DIR,'kegg','approved_example.txt'),
                'our':os.path.join(RAW_DATA_DIR,'our_multi','approved_example.txt')}

SMILE_FILE = {
               'kegg': os.path.join(RAW_DATA_DIR,'kegg', 'smiles.npy'),
                'our': os.path.join(RAW_DATA_DIR,'our_multi', 'smiles.npy')
                }

SEPARATOR = {'kegg':'\t', 'our':'\t' }
SEPARATOR_KG = {'kegg':' ', 'our':'\t' }
THRESHOLD = {'our':4,'kegg':4} 
NEIGHBOR_SIZE = {'kegg':16,'our':1}

#
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
RESULT_LOG={'kegg':'kegg_result.txt', 'our': 'our_result.txt'}
PERFORMANCE_LOG = 'performance.log'
DRUG_EXAMPLE='{dataset}_examples.npy'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 4 
        self.embed_dim = 32 
        self.n_depth = 2    
        self.l2_weight = 1e-7  
        self.lr = 2e-2  
        self.batch_size = 0
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.inter_types = 1
        self.optimizer = 'adagrad'
        self.len_train = None
        self.drug_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None
        self.embed_smile = None
        self.len_smile = 0

        self.exp_name = None
        self.model_name = None

        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_f1_macro'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        self.early_stopping_monitor = 'val_f1_macro'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='kegg'
        self.K_Fold=1
        self.callbacks_to_add = None
        self.swa_start = 3

