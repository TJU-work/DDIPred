import sys
import random
import os
import numpy as np
from collections import defaultdict

sys.path.append(os.getcwd()) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from main import train
from config import SEPARATOR_KG, DRUG_EXAMPLE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ENTITY2ID_FILE, KG_FILE, \
    EXAMPLE_FILE,  DRUG_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, THRESHOLD, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE, SMILE_FILE
from utils import pickle_dump, format_filename, write_log


def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict):
    print(f'Logging Info - Reading entity2id file: {file_path}' )
    assert len(drug_vocab) == 0 and len(entity_vocab) == 0
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if(count==0):
                count+=1
                continue
            drug, entity = line.strip().split('\t')
            drug_vocab[entity]=len(drug_vocab) 
            entity_vocab[entity] = len(entity_vocab)

def read_example_file(file_path:str,separator:str,drug_vocab:dict):
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(drug_vocab)>0
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            d1,d2,flag=line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1],drug_vocab[d2],int(flag)])
    
    examples_matrix=np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    X=examples_matrix[:,:2]
    y=examples_matrix[:,2:3]
    train_data_X, valid_data_X,train_y,val_y = train_test_split(X,y, test_size=0.2,stratify=y)
    train_data=np.c_[train_data_X,train_y]
    valid_data_X, test_data_X,val_y,test_y = train_test_split(valid_data_X,val_y, test_size=0.5)
    valid_data=np.c_[valid_data_X,val_y]
    test_data=np.c_[test_data_X,test_y]
    return examples_matrix

def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int, separator:str):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if count==0:
                count+=1
                continue
            head, tail, relation = line.strip().split(separator) 

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True
        )

        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation

def read_smiles_pretrain(dataset : str):
    if dataset=='kegg':
        return np.load(SMILE_FILE[dataset], allow_pickle=True).item()
    return np.load(SMILE_FILE[dataset], allow_pickle=True)

def count_interaction_types(matrics:list):
    inte_types = set()
    for row in matrics:
        inte_types.add(row[2])
    return len(inte_types)

def process_data(dataset: str, neighbor_sample_size: int,K:int):
    drug_vocab = {}
    entity_vocab = {}
    relation_vocab = {}

    read_entity2id_file(ENTITY2ID_FILE[dataset], drug_vocab, entity_vocab)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),drug_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),entity_vocab)

    examples_file = format_filename(PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset)
    examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset],drug_vocab)
    interactions_types = count_interaction_types(examples)
    np.save(examples_file,examples)
          
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    
    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size, SEPARATOR_KG[dataset])
    
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),
                drug_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)
    
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_entity_file)

    smiles_pretraining = read_smiles_pretrain(dataset)

    cross_validation(K,examples,dataset,neighbor_sample_size, smiles_pretraining, interactions_types)


def cross_validation(K_fold,examples,dataset,neighbor_sample_size, smiles, inter_types):

    subsets=dict()
    n_subsets=int(len(examples)/K_fold)
    remain=set(range(0,len(examples)-1))
    for i in reversed(range(0,K_fold-1)):
        subsets[i]=random.sample(remain,n_subsets)
        remain=remain.difference(subsets[i])
    subsets[K_fold-1]=remain
    

    aggregator_types=['single']
    for t in aggregator_types:
        count=1
        temp={'dataset':dataset,'aggregator_type':t,'avg_auc':0.0,'avg_acc':0.0,'avg_f1':0.0,'avg_aupr':0.0, 'avg_rec': 0.0, 'avg_pre':0.0}
        
        for i in reversed(range(0,K_fold)):
            test_d=examples[list(subsets[i])]
            val_d,test_data=train_test_split(test_d,test_size=0.5)
            train_d=[]
            for j in range(0,K_fold):
                if i!=j:
                    train_d.extend(examples[list(subsets[j])])
            train_data=np.array(train_d)
 
            train_log=train(
            kfold=count,
            dataset=dataset,
            train_d=train_data,
            dev_d=val_d,
            test_d=test_data,
            neighbor_sample_size=neighbor_sample_size,
            embed_dim=128,
            n_depth=1,
            l2_weight=1e-7,
            lr=2e-2,
            optimizer_type='adam',
            batch_size=2048,
            aggregator_type=t,
            n_epoch=200,
            callbacks_to_add=['modelcheckpoint', 'earlystopping'],
            embed_smile=smiles
            )     
            count+=1
            temp['avg_auc']=temp['avg_auc']+train_log['test_auc']
            temp['avg_acc']=temp['avg_acc']+train_log['test_acc']
            temp['avg_f1']=temp['avg_f1']+train_log['test_f1']
            temp['avg_aupr']=temp['avg_aupr']+train_log['test_aupr']
            temp['avg_rec']=temp['avg_rec']+train_log['test_rec']
            temp['avg_pre']=temp['avg_pre']+train_log['test_pre']
        for key in temp:
            if key=='aggregator_type' or key=='dataset':
                continue
            temp[key]=temp[key]/K_fold
        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]),temp,'a')
        print(f'Logging Info - 5 Fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}, avg_rec: {temp["avg_rec"]}, avg_pre: {temp["avg_pre"]}')
   
if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    dataset = 'our'
    process_data(dataset,NEIGHBOR_SIZE[dataset],5)
