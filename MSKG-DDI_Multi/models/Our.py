from distutils.command.config import config
from turtle import shape
from unicodedata import name
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras import backend as K  
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
import sklearn.metrics as m
from layers import Aggregator
from callbacks import Metric
from models.base_model import BaseModel
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.losses import losses_utils
import tensorflow as tf
import numpy as np


class Our(BaseModel):
    def __init__(self, config):
        super(Our, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            shape=(1,), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1, ), name='input_drug_two', dtype='int64')

        input_smiles_one = Input(shape=(self.config.len_smile, ), name='input_smiles_one', dtype='float32' )
        input_smiles_two = Input(shape=(self.config.len_smile, ), name='input_smiles_two', dtype='float32')
        
        drug_one_embedding = Embedding(input_dim=self.config.drug_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='user_embedding')
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')

        drug_embed = drug_one_embedding(
            input_drug_one)  # [batch_size, 1, embed_dim]

        receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x),
                                         name='receptive_filed_drug_one')(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth+1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth+1:]

        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding_drug_one')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}_drug_one'
            )
        
            next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed = neighbor_embedding([drug_embed, neigh_rel_embed_list_drug_one[hop],
                                                     neigh_ent_embed_list_drug_one[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list_drug_one[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one

        # get receptive field
        receptive_list = Lambda(lambda x: self.get_receptive_field(x),
                                name='receptive_filed')(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth+1]
        neigh_rel_list = receptive_list[self.config.n_depth+1:]

        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]

        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth+1}'
            )

            next_neigh_ent_embed_list = []
            for hop in range(self.config.n_depth-depth):
                neighbor_embed = neighbor_embedding([drug_embed, neigh_rel_embed_list[hop],
                                                     neigh_ent_embed_list[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list[hop], neighbor_embed])
                next_neigh_ent_embed_list.append(next_entity_embed)
            neigh_ent_embed_list = next_neigh_ent_embed_list

        drug1_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list_drug_one[0])
        drug2_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list[0])

        drug1_squeeze_embed_all  = Lambda( lambda x: K.concatenate(x, axis=1))([drug1_squeeze_embed,input_smiles_one])     
        layer_f11 = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(drug1_squeeze_embed_all)
        layer_f12 = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(layer_f11)
        drug1_squeeze_embed_all = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(layer_f12)


        drug2_squeeze_embed_all  = Lambda( lambda x: K.concatenate(x, axis=1))([drug2_squeeze_embed,input_smiles_two])
        layer_f21 = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(drug2_squeeze_embed_all)
        layer_f22 = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(layer_f21)
        drug2_squeeze_embed_all = Sequential([Dense(100,activation=None),BatchNormalization(momentum=0.1), LeakyReLU(alpha=0.1)])(layer_f22)

        concat = Lambda( lambda x: K.concatenate(x, axis=1))([drug1_squeeze_embed_all, drug2_squeeze_embed_all])
        layer1 = Sequential([Dense(2048,activation=None),BatchNormalization(momentum=0.1),Activation('relu')])(concat)
        layer2 = Sequential([Dense(2048,activation=None),BatchNormalization(momentum=0.1),Activation('relu')])(layer1) 
        score = Dense(self.config.inter_types,activation='softmax')(layer2)

        loss_config = CategoricalCrossentropy()

        model = Model([input_drug_one, input_drug_two, input_smiles_one, input_smiles_two], score)
        model.compile(optimizer=self.config.optimizer,
                      loss=loss_config, metrics=['acc'])
        return model

    def get_receptive_field(self, entity):
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = K.sum(drug * rel, axis=-1, keepdims=True)

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed


    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(Metric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, 
                                         self.config.K_Fold, self.config.batch_size))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_pred = self.model.predict(x)

        y_true = y
        y_true_labels = np.zeros((len(y_true,)))
        for i,out in enumerate(y_true):
            y_true_labels[i] = np.argmax(out)

        y_pred_labels = np.zeros((len(y_pred,)))
        for i,out in enumerate(y_pred):
            y_pred_labels[i] = np.argmax(out)

        try:
            auc = roc_auc_score(y_true=y_true_labels, y_score=y_pred, average='macro', multi_class='ovo')
        except:
            auc = 0
        try:
            aupr = self.roc_aupr_score(y_true, y_pred, average='macro')
        except:
            aupr = 0

        acc = accuracy_score(y_true=y_true_labels, y_pred=y_pred_labels)
        f1_macro = f1_score(y_true=y_true_labels, y_pred=y_pred_labels, average='macro')
        recall_macro = recall_score(y_true=y_true_labels, y_pred=y_pred_labels,average='macro')
        precision_macro = precision_score(y_true=y_true_labels, y_pred=y_pred_labels,average='macro')

        return auc, acc, f1_macro, aupr, precision_macro, recall_macro

    def roc_aupr_score(self,y_true, y_score, average="macro"):
        def _binary_roc_aupr_score(y_true, y_score):
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
            return m.auc(recall, precision)

        def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
            if average == "binary":
                return binary_metric(y_true, y_score)
            if average == "micro":
                y_true = y_true.ravel()
                y_score = y_score.ravel()
            if y_true.ndim == 1:
                y_true = y_true.reshape((-1, 1))
            if y_score.ndim == 1:
                y_score = y_score.reshape((-1, 1))
            n_classes = y_score.shape[1]
            score = np.zeros((n_classes,))
            for c in range(n_classes):
                y_true_c = y_true.take([c], axis=1).ravel()
                y_score_c = y_score.take([c], axis=1).ravel()
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)
        return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
    
