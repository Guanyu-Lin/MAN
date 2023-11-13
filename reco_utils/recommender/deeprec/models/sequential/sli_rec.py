# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import (
    SequentialBaseModel,
)
from tensorflow.nn import dynamic_rnn
from reco_utils.recommender.deeprec.models.sequential.rnn_cell_implement import (
    Time4LSTMCell,
)

__all__ = ["SLI_RECModel"]


class SLI_RECModel(SequentialBaseModel):
    """SLI Rec model

    Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, "Adaptive User Modeling with
    Long and Short-Term Preferences for Personailzed Recommendation", in Proceedings of 
    the 28th International Joint Conferences on Artificial Intelligence, IJCAI’19, 
    Pages 4213-4219, AAAI Press, 2019.
    """

    def _build_seq_graph(self):
        """The main function to create sli_rec model.
        
        Returns:
            obj:the output of sli_rec section.
        """
        hparams = self.hparams
        with tf.variable_scope("sli_rec"):
            hist_input = tf.concat(
                [self.item_history_embedding, self.cate_history_embedding], 2
            )
            self.mask = self.iterator.mask
            self.sequence_length = tf.reduce_sum(self.mask, 1)

            with tf.variable_scope("long_term_asvd"):
                att_outputs1 = self._attention(hist_input, hparams.attention_size)
                att_fea1 = tf.reduce_sum(att_outputs1, 1)
                tf.summary.histogram("att_fea1", att_fea1)

            item_history_embedding_new = tf.concat(
                [
                    self.item_history_embedding,
                    tf.expand_dims(self.iterator.time_from_first_action, -1),
                ],
                -1,
            )
            item_history_embedding_new = tf.concat(
                [
                    item_history_embedding_new,
                    tf.expand_dims(self.iterator.time_to_now, -1),
                ],
                -1,
            )
            with tf.variable_scope("rnn"):
                rnn_outputs, final_state = dynamic_rnn(
                    Time4LSTMCell(hparams.hidden_size),
                    inputs=item_history_embedding_new,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32,
                    scope="time4lstm",
                )
                tf.summary.histogram("LSTM_outputs", rnn_outputs)

            with tf.variable_scope("attention_fcn"):
                att_outputs2 = self._attention_fcn(
                    self.target_item_embedding, rnn_outputs
                )
                att_fea2 = tf.reduce_sum(att_outputs2, 1)
                tf.summary.histogram("att_fea2", att_fea2)

            # ensemble
            with tf.name_scope("alpha"):
                if not hparams.manual_alpha:
                    concat_all = tf.concat(
                        [
                            self.target_item_embedding,
                            att_fea1,
                            att_fea2,
                            tf.expand_dims(self.iterator.time_to_now[:, -1], -1),
                        ],
                        1,
                    )
                    last_hidden_nn_layer = concat_all
                    alpha_logit = self._fcn_net(
                        last_hidden_nn_layer, hparams.att_fcn_layer_sizes, scope="fcn_alpha"
                    )
                    self.alpha_output = tf.sigmoid(alpha_logit)
                    user_embed = att_fea1 * self.alpha_output + att_fea2 * (1.0 - self.alpha_output)
                    tf.summary.histogram("alpha", self.alpha_output)
                    error_with_category = self.alpha_output - self.iterator.attn_labels
                    tf.summary.histogram("error_with_category", error_with_category)
                    squared_error_with_category = tf.math.sqrt(tf.math.squared_difference(tf.reshape(self.alpha_output, [-1]), tf.reshape(self.iterator.attn_labels, [-1])))
                    tf.summary.histogram("squared_error_with_category", squared_error_with_category)
                else:
                    user_embed = att_fea1 * hparams.manual_alpha_value + att_fea2 * (1.0 - hparams.manual_alpha_value)
            model_output = tf.concat([user_embed, self.target_item_embedding], 1)
            tf.summary.histogram("model_output", model_output)
            return model_output
