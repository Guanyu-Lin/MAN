# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from reco_utils.recommender.deeprec.models.sequential.sli_rec import (
    SLI_RECModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import VecAttGRUCell
from reco_utils.recommender.deeprec.models.sequential.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
from tensorflow.contrib.rnn import GRUCell, LSTMCell

__all__ = ["SASRecModel"]


class SASRecModel(SLI_RECModel):
    def _encoder(self, item_history_embedding, cate_history_embedding, position_embedding, domain):
        with tf.variable_scope('sasrec%s'%domain):
            seq = tf.concat(
                [item_history_embedding, cate_history_embedding], 2
            )
            seq = seq + position_embedding
            mask = self.iterator.mask
            real_mask = tf.cast(mask, tf.float32)
            self.sequence_length = tf.reduce_sum(mask, 1)
            #  attention_output = self._attention_fcn(self.target_item_embedding, hist_input)
            #  att_fea = tf.reduce_sum(attention_output, 1)
            # hyper-parameters
            self.dropout_rate = 0.0
            self.num_blocks = 2
            self.hidden_units = self.item_embedding_dim + self.cate_embedding_dim
            self.num_heads = 1
            self.is_training = True
            #  self.recent_k = 5
            self.recent_k = 1

            # Dropout
            #  self.seq = tf.layers.dropout(self.seq,
                                         #  rate=self.dropout_rate,
                                         #  training=tf.convert_to_tensor(self.is_training))
            seq *= tf.expand_dims(real_mask, -1)
            local_PE_emb = seq
            # Build blocks
            # 堆叠的注意力，参数为2
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

   
                    # Self-attention
                    seq = self.multihead_attention(queries=self.normalize(seq),
                                                     keys=seq,
                                                     num_units=self.hidden_units,
                                                     num_heads=self.num_heads,
                                                     dropout_rate=self.dropout_rate,
                                                     is_training=self.is_training,
                                                     causality=True,
                                                     #  causality=False,
                                                     scope=domain)


                    # Feed forward
                    seq = self.feedforward(self.normalize(seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training)
                    seq *= tf.expand_dims(real_mask, -1)


            seq = self.normalize(seq)

            # all 
            #  self.hist_embedding_sum = tf.reduce_sum(self.seq*tf.expand_dims(self.real_mask, -1), 1)
            hist_embedding_mean = tf.reduce_sum(seq*tf.expand_dims(real_mask, -1), 1)/tf.reduce_sum(real_mask, 1, keepdims=True)

        return seq, local_PE_emb, hist_embedding_mean, real_mask


    def _build_seq_graph(self):
     
        self.local_seqA, self.local_PE_embA, hist_embedding_mean_A, real_maskA = self._encoder(self.item_history_embedding_A, self.cate_history_embedding_A, self.position_embedding_A, "A")
        self.local_seqB, self.local_PE_embB, hist_embedding_mean_B, real_maskB = self._encoder(self.item_history_embedding_B, self.cate_history_embedding_B, self.position_embedding_B, "B")
        self.global_seq, self.global_PE_emb, hist_embedding_mean, real_mask = self._encoder(self.item_history_embedding, self.cate_history_embedding, self.position_embedding, "global")

        self.cross_seqA = self._cross_choronological_modeling(self.local_seqA, self.global_seq, real_maskA, "A")
        self.cross_seqB = self._cross_choronological_modeling(self.local_seqB, self.global_seq, real_maskB, "B")
        print("I am sasrec")
        self.cross_item_A = self._cross_itemSimilarity_modeling(self.local_PE_embA, self.global_PE_emb, "A")
        self.cross_item_B = self._cross_itemSimilarity_modeling(self.local_PE_embB, self.global_PE_emb, "B")

        self.group_A, attn_weight_A = self._cross_group_modeling(self.local_seqA, "A")
        self.group_B, attn_weight_B = self._cross_group_modeling(self.local_seqB, "B")

        cross_seqA_mean = tf.reduce_sum(self.cross_seqA*tf.expand_dims(real_maskA, -1), 1)/tf.reduce_sum(real_maskA, 1, keepdims=True)
        cross_seqB_mean = tf.reduce_sum(self.cross_seqB*tf.expand_dims(real_maskB, -1), 1)/tf.reduce_sum(real_maskB, 1, keepdims=True)

        model_outputA = tf.concat([self.target_item_embedding, self.cross_item_A , hist_embedding_mean_A, cross_seqA_mean, self.group_A, hist_embedding_mean], -1)
    
        model_outputB = tf.concat([self.target_item_embedding, self.cross_item_B, hist_embedding_mean_B, cross_seqB_mean, self.group_B, hist_embedding_mean], -1)


        return model_outputA, model_outputB, attn_weight_A, attn_weight_B

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

    def _attention_fcn(self, query, key_value, name, reuse, mask, return_alpha=False):
        
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W 
            reuse (obj): Reusing variable W in query operation 
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.variable_scope("attention_fcn"+str(name), reuse=reuse):
            query_size = query.shape[-1].value
            boolean_mask = tf.equal(mask, tf.ones_like(mask))

            attention_mat = tf.get_variable(
                name="attention_mat"+str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            )
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])

            if query.shape.ndims != att_inputs.shape.ndims:
                queries = tf.reshape(
                    tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
                )
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1)
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            # print("mask", boolean_mask.shape.as_list())
            # print("attn out", att_fnc_output.shape.as_list())
            # print("mask_padidng", mask_paddings.shape.as_list())

            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = key_value * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights
