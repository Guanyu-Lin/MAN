# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import socket
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from tensorflow.keras import backend as K
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.nn import dynamic_rnn

from tensorflow.keras import backend as K

__all__ = ["SURGEModel"]


class SURGEModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.relative_threshold = 0.1 # default
        # self.relative_threshold = 0.3
        self.num_heads = 1
        self.metric_heads = 1
        self.attention_heads = 1
        self.dropout_rate = 0.0
        self.is_training = True

        self.pool_layers = 1
        self.layer_shared = True
        self.remove_target = False
        self.recent_target = False
        # graph loss
        self.smoothness_ratio = 0.1
        self.degree_ratio = 0.1
        self.sparsity_ratio = 0.1

        self.same_mapping_regu, self.single_affiliation_regu, self.relative_position_regu = 1e-2, 1e-2, 1e-3

        #  self.pool_ratio = 0.5
        # self.pool_ratio = 0.1
        self.pool_ratio = 0.6 # default

        self.pool_length = 10 # 
        super().__init__(hparams, iterator_creator, seed=None)
    def _build_seq_graph(self):
            self.local_seqA, self.local_PE_embA, final_stateA,  graph_readoutA, real_maskA, self.auxiliary_lossA = self._encoder(self.item_history_embedding_A, self.cate_history_embedding_A, self.position_embedding_A, "A")
            self.local_seqB, self.local_PE_embB, final_stateB,  graph_readoutB, real_maskB, self.auxiliary_lossB = self._encoder(self.item_history_embedding_B, self.cate_history_embedding_B, self.position_embedding_B, "B")
            self.global_seq, self.global_PE_emb, final_state,  graph_readout, real_mask, self.auxiliary_loss_global = self._encoder(self.item_history_embedding, self.cate_history_embedding, self.position_embedding, "global")

            self.cross_seqA = self._cross_choronological_modeling(self.local_seqA, self.global_seq, real_maskA, "A")
            self.cross_seqB = self._cross_choronological_modeling(self.local_seqB, self.global_seq, real_maskB, "B")

            self.cross_item_A = self._cross_itemSimilarity_modeling(self.local_PE_embA, self.global_PE_emb, "A")
            self.cross_item_B = self._cross_itemSimilarity_modeling(self.local_PE_embB, self.global_PE_emb, "B")

            self.group_A, attn_weight_A = self._cross_group_modeling(self.local_seqA, "A")
            self.group_B, attn_weight_B = self._cross_group_modeling(self.local_seqB, "B")



            model_outputA = tf.concat([final_stateA, graph_readoutA, self.target_item_embedding_A, graph_readoutA*self.target_item_embedding_A, self.cross_item_A, self.group_A, self.cross_seqA[:, -1, :]], 1)
            model_outputB = tf.concat([final_stateB, graph_readoutB, self.target_item_embedding_B, graph_readoutB*self.target_item_embedding_B, self.cross_item_B, self.group_B, self.cross_seqB[:, -1, :]], 1)
            model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout*self.target_item_embedding], 1)
            return model_outputA, model_outputB, model_output, attn_weight_A, attn_weight_B




    def _encoder(self, item_history_embedding, cate_history_embedding, position_embedding, domain):
        with tf.variable_scope('Encoder%s'%domain):

            X = tf.concat(
                [item_history_embedding, cate_history_embedding], 2
            )
            X = X + position_embedding
            local_PE_emb = X
            mask = self.iterator.mask

            self.float_mask = tf.cast(mask, tf.float32)
            self.real_sequence_length = tf.reduce_sum(mask, 1)
            # compute recent for matching
            self.recent_k = 1
            real_mask = tf.cast(mask, tf.float32)
            self.position = tf.math.cumsum(real_mask, axis=1, reverse=True)
            self.recent_mask = tf.logical_and(self.position >= 1, self.position <= self.recent_k)
            self.real_recent_mask = tf.where(self.recent_mask, tf.ones_like(self.recent_mask, dtype=tf.float32), tf.zeros_like(self.recent_mask, dtype=tf.float32))
            self.recent_embedding_mean = tf.reduce_sum(X*tf.expand_dims(self.real_recent_mask, -1), 1)/tf.reduce_sum(self.real_recent_mask, 1, keepdims=True)

            self.max_n_nodes = int(X.get_shape()[1])
            
            with tf.variable_scope('interest_graph'):

                S = []
    
                for i in range(self.metric_heads):
                    # weighted cosine similarity
                    self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False)
                    X_fts = X * tf.expand_dims(self.weighted_tensor, 0)
                    X_fts = tf.nn.l2_normalize(X_fts,dim=2)
                    S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # B*L*L


                    S += [S_one]
                S = tf.reduce_mean(tf.stack(S, 0), 0)
                S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)


                # mask invalid nodes


                ## Graph sparsification via seted sparseness
                S_flatten = tf.reshape(S, [tf.shape(S)[0],-1])
                
                sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
                    
                # relative ranking strategy of the entire graph
                num_edges = tf.cast(tf.count_nonzero(S, [1,2]), tf.float32) # B

                to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)

                threshold_score = tf.gather_nd(sorted_S_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
                A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32)

            with tf.variable_scope('interest_fusion_extraction'):
                for l in range(self.pool_layers):
                    reuse = False if l==0 else True

                    X, A, graph_readout, alphas, tmp_mask, auxiliary_loss = self._interest_fusion_extraction_new(X, A, mask, layer=l, reuse=reuse)

                    #  X = self.normalize(X)
            mask = tmp_mask
            real_mask = tf.cast(mask, tf.float32)
            # print("before X", X.shape.as_list())

            with tf.variable_scope('encoder'):

                self.reduced_sequence_length = tf.reduce_sum(mask, 1) # B

                output_seq_state, final_state = dynamic_rnn_dien(
                    VecAttGRUCell(self.hparams.hidden_size),
                    inputs=X,
                    att_scores = tf.expand_dims(alphas, -1),
                    sequence_length=self.reduced_sequence_length,
                    dtype=tf.float32,
                    scope="gru"
                )



        return output_seq_state, local_PE_emb, final_state,  graph_readout, real_mask, auxiliary_loss


  


    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling 

        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation 

        Returns:
            X (obj): Aggerated cluster embedding 
            A (obj): Pooled adjacency matrix 
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer

        """
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            Xq = tf.matmul(A, tf.matmul(A, X)) # B*L*F

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                ## graph attentive convolution
                if not self.remove_target:
                    E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                else:
                    E = A_bool * tf.expand_dims(f_1,1) 
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, 32, use_bias=False)
                #  Xc_one = self.normalize(Xc_one)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

            #  Xc = self.normalize(Xc)

        with tf.name_scope('interest_extraction'):
            ## cluster fitness score 
            Xq = tf.matmul(A, tf.matmul(A, Xc)) # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True)
                if not self.remove_target:
                    cluster_score += [f_1 + f_2]
                else:
                    cluster_score += [f_1]
            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0)
            boolean_mask = tf.equal(mask, tf.ones_like(mask))
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(
                tf.where(boolean_mask, cluster_score, mask_paddings),
                axis = -1
            )

            ## graph pooling
            num_nodes = tf.reduce_sum(mask, 1) # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool, 
                               tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32), 
                               num_nodes)  # B
            cluster_score = cluster_score * self.float_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1) # B*2
                target_score = tf.gather_nd(sorted_score, target_index) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(mask, 1)

            ## ensure graph connectivity 
            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool),
                          tf.transpose(E, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout 
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.float_mask, -1), 1)

        return Xc, A, graph_readout, cluster_score

    def _interest_fusion_extraction_new(self, X, A, mask, layer, reuse):
        with tf.name_scope('interest_fusion'):
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1])
            D = tf.reduce_sum(A_bool, axis=-1) # B*L
            D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            Xq = tf.matmul(A, tf.matmul(A, X)) # B*L*F

            # multi-head gat
            Xc = []
            node_score = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_layer_'+str(layer)+'_'+str(i), False, mask, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, mask, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_layer_'+str(layer)+'_'+str(i), False, mask, return_alpha=True)
                if self.layer_shared:
                    _, f_1 = self._attention_fcn(Xq, X, 'f1_shared'+'_'+str(i), reuse, mask, return_alpha=True)
                    if not self.remove_target:
                        if not self.recent_target:
                            _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, mask, return_alpha=True)
                        else:
                            _, f_2 = self._attention_fcn(self.recent_embedding_mean, X, 'f2_shared'+'_'+str(i), reuse, mask, return_alpha=True)

                ## graph attentive convolution
                if not self.remove_target:
                    E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                    
                else:
                    E = A_bool * tf.expand_dims(f_1,1)
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                Xc_one = tf.layers.dense(Xc_one, self.hparams.hidden_size, use_bias=False)
                #  Xc_one = self.normalize(Xc_one)
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
                #  Xc += [Xc_one]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0)

            node_score = f_2
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(node_score,-1)*tf.expand_dims(tf.cast(mask, tf.float32), -1), 1)

            Xc, A, cluster_score, mask, auxiliary_loss = self._diffpool(Xc, A, node_score, mask)


        return Xc, A, graph_readout, cluster_score, mask, auxiliary_loss
        #  return Xc, A, graph_readout, node_score
    def gat(self, X, num_heads, A_bool):

        #  X = self.normalize(X)

        Q = tf.layers.conv1d(X, 32, 1) # B*L*F
        K = tf.layers.conv1d(X, 32, 1)
        V = tf.layers.conv1d(X, 32, 1, use_bias=False)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0) # B*LxH*F/H
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

        att_score_ = tf.reduce_sum(Q_ * K_, axis=-1) # B*LxH

        A_bool_ = tf.tile(A_bool, [1, num_heads, 1]) # B*L*L -> B*LxH*L
        E = A_bool_ * tf.expand_dims(att_score_,-1)  # B*LxH*L x B*LxH*1 -> B*LxH*L
        E = tf.nn.leaky_relu(E)
        boolean_mask_ = tf.equal(A_bool_, tf.ones_like(A_bool_))
        mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1) # B*LxH*L
        E = tf.nn.softmax(
            tf.where(boolean_mask_, E, mask_paddings), # B*LxH*L
            axis = -1
        )
        #  Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
        h_ = tf.matmul(E, V_) # B*LxH*L x B*L*F/H -> B*LxH*F/H
        #  Xc_one = tf.layers.dense(Xc_one, 32, use_bias=False)

        h = tf.concat(tf.split(h_, num_heads, axis=0), axis=-1) # B*L*F

        h += X
        h = tf.nn.leaky_relu(h)

        h = self.normalize(h)

        return h, E


    def normalize(self, inputs, 
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.
        
        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
          
        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
            
        return outputs

    def _compute_graph_loss(self):
        """Graph regularization loss"""

        #  import pdb; pdb.set_trace()
        L = tf.ones_like(self.A) * tf.eye(self.max_n_nodes) * tf.reduce_sum(self.A, -1, keep_dims=True) - self.A

        #  laplacian（效果稍好于上面，但差不多）
        #  D = tf.reduce_sum(self.A, axis=-1) # B*L
        #  D = tf.sqrt(D)[:, None] + K.epsilon() # B*1*L
        #  L = (self.A / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1

        graph_loss = self.smoothness_ratio * tf.trace(tf.matmul(tf.transpose(self.X, (0,2,1)), tf.matmul(L, self.X))) / (self.max_n_nodes*self.max_n_nodes)
        ones_vec = tf.tile(tf.ones([1,self.max_n_nodes]), [tf.shape(self.A)[0], 1]) # B*L
        graph_loss += -self.degree_ratio * tf.squeeze(tf.matmul(tf.expand_dims(ones_vec, 1), tf.log(tf.matmul(self.A, tf.expand_dims(ones_vec,-1)) + K.epsilon())), (-1,-2)) / self.max_n_nodes
        graph_loss += self.sparsity_ratio * tf.reduce_sum(tf.math.pow(self.A, 2), (-1,-2)) / self.max_n_nodes
        graph_loss = tf.reduce_mean(graph_loss)

        return graph_loss

    # def _get_loss(self):
    #     """Make loss function, consists of data loss, regularization loss and graph loss

    #     Returns:
    #         obj: Loss value
    #     """

    #     self.data_loss = self._compute_data_loss()
    #     self.regular_loss = self._compute_regular_loss()



    #     self.loss = self.data_loss + self.regular_loss + self.auxiliary_loss


    #     #  self.loss = self.data_loss + self.regular_loss + self.graph_loss + self.auxiliary_loss

    #     return self.loss

    def _get_lossA(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_lossA = self._compute_data_lossA()
        self.disentangle_lossA =tf.reduce_sum( self._disentangle_loss())
        self.regular_loss = self._compute_regular_loss()
        # print(self.disentangle_lossA)
        # print(self.data_lossA)
        #self.lossA = self.data_lossA
        # self.lossA = tf.add(tf.add(self.data_lossA, self.disentangle_lossA), self.auxiliary_loss)
        self.lossA = self.data_lossA + self.auxiliary_lossA + self.regular_loss 
        # self.lossA = (self.data_lossA)

        return self.lossA


    def _get_lossB(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_lossB = self._compute_data_lossB()
        self.disentangle_lossB =tf.reduce_sum( self._disentangle_loss())
        # print(self.disentangle_lossB)
        # print(self.data_lossB)
        self.regular_loss = self._compute_regular_loss()

        #self.lossA = self.data_lossA
        # self.lossB = tf.add(tf.add(self.data_lossB, self.disentangle_lossB), self.auxiliary_loss)
        self.lossB = self.data_lossB + self.auxiliary_lossB + self.regular_loss + self.disentangle_lossB 
        return self.lossB

    def _diffpool(self, X, A, node_score, mask):

        hparams = self.hparams
        with tf.name_scope('diffpool'):
            k = 30
            for _ in range(self.pool_layers):
                # Update node embeddings
                Z = tf.layers.dense(X, 32, use_bias=False)

                # Compute cluster assignment matrix
                S = tf.layers.dense(X, k, use_bias=False) # B*L*F -> B*L*k
                S = tf.matmul(A, S) # B*L*L x B*L*k -> B*L*k
 
 
                # 2.via ratio
                num_nodes = tf.cast(tf.reduce_sum(mask, 1), tf.float32) # B
                node_position = tf.cast(tf.math.cumsum(mask, axis=1), tf.float32) # B*L 
                boolean_pool = tf.less(node_position, tf.expand_dims(self.pool_ratio*num_nodes, -1)) # B*L + B*1 -> B*L(k)
                # if k not 50
                boolean_pool = tf.batch_gather(boolean_pool, tf.tile(tf.expand_dims(tf.range(k), 0), [tf.shape(mask)[0], 1]))

                mask_paddings = tf.ones_like(S) * (-(2 ** 32) + 1) # B*L*L(k)
                S = tf.nn.softmax(
                    tf.where(tf.tile(tf.expand_dims(boolean_pool, 1), [1, tf.shape(mask)[1], 1]), S, mask_paddings),
                    axis = -1
                ) # B*1*L(k) + B*L*L(k) + B*L*(k) -> B*L*L(k)

                mask = tf.cast(boolean_pool, tf.int32) # B*L(k)

                ## Auxiliary pooling loss
                auxiliary_loss = 0.0
                # Link prediction loss
                S_gram = tf.matmul(S, tf.transpose(S, (0,2,1)))
                LP_loss = A - S_gram #  LP_loss = A/tf.norm(A) - S_gram/tf.norm(S_gram)
                LP_loss = tf.norm(LP_loss, axis=(-1, -2))
                LP_loss = K.mean(LP_loss)
                auxiliary_loss += self.same_mapping_regu*LP_loss
                # Entropy loss
                entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
                entr_loss = K.mean(entr, axis=-1)
                entr_loss = K.mean(entr_loss)
                auxiliary_loss += self.single_affiliation_regu*entr_loss
                # Position loss
                Pn = tf.math.cumsum(tf.ones([tf.shape(S)[0], 1, tf.shape(S)[1]]), axis=-1) # node position encoding:
                Pc = tf.math.cumsum(tf.ones([tf.shape(S)[0], 1, tf.shape(S)[2]]), axis=-1) # cluster position encoding:
                position_loss = tf.matmul(Pn, S) - Pc
                position_loss = tf.norm(position_loss, axis=(-1, -2))
                position_loss = K.mean(position_loss)
                auxiliary_loss += self.relative_position_regu*position_loss

                # pooled
                X = tf.matmul(tf.transpose(S, (0,2,1)), Z) # B*L(k)*L x B*L*F -> B*L(k)*F
                cluster_score = tf.squeeze(tf.matmul(tf.transpose(S, (0,2,1)), tf.expand_dims(node_score, -1)), -1) # B*L(k)*L x B*L*1 -> B*L(k)*1 -> B*L(k)


                X = X*tf.expand_dims(cluster_score, -1)
                A = tf.matmul(
                    tf.matmul(
                        tf.transpose(S, (0,2,1)), 
                        A), 
                    S) # B*L(k)*L x B*L*L x B*L*L(k) -> B*L(k)*L(k)


        return X, A, cluster_score, mask, auxiliary_loss

