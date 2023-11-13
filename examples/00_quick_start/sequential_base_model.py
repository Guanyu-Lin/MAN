# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import socket
import pdb
from reco_utils.recommender.deeprec.models.base_model import BaseModel
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

__all__ = ["SequentialBaseModel"]


class SequentialBaseModel(BaseModel):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams
        self.step_A = 0
        self.step_B = 0
        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams else 1
        )
        self.hidden_size = hparams.hidden_size if "hidden_size" in hparams else None
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass
    def _encoder(self, domain):
        """Subclass will implement this."""
        pass
    def _cross_choronological_modeling(self, local_seq, global_seq, real_mask, domain):
        with tf.variable_scope("cross_attention_%s"%domain):

            seq_cross = self.multihead_attention(queries=self.normalize(tf.stop_gradient(local_seq), scope=domain),
                                                keys=tf.stop_gradient(global_seq),
                                                num_units=self.hparams.hidden_size,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=True,
                                                #  causality=False,
                                                scope=domain)
                                                
            seq_cross = self.feedforward(self.normalize(seq_cross, scope=domain), num_units=[self.hparams.hidden_size, self.hparams.hidden_size],                                           dropout_rate=self.dropout_rate, is_training=self.is_training, scope=domain)
            seq_cross *= tf.expand_dims(real_mask, -1)
        return seq_cross

    def _cross_itemSimilarity_modeling(self, local_PE_emb, global_PE_emb, domain):

        attention_output, alphas = self._attention_fcn(self.target_item_embedding, local_PE_emb + global_PE_emb, 'Att_%s'%domain, False, self.iterator.mask, return_alpha=True)
        att_fea = tf.reduce_sum(attention_output, 1)

        return att_fea
    def _cross_group_modeling(self, local_seq, domain):

        with tf.variable_scope("group_attention_%s"%domain):

            seq_group = self.multihead_attention(queries=self.normalize(self.user_group_emb, scope='construction'),
                                keys=tf.stop_gradient(local_seq),
                                num_units=self.hparams.hidden_size,
                                num_heads=self.num_heads,
                                dropout_rate=self.dropout_rate,
                                is_training=self.is_training,
                                causality=True,
                                #  causality=False,
                                scope='groupConstruction_%s'%domain)
            seq_group= self.feedforward(self.normalize(seq_group, scope='construction_group_%s'%domain), num_units=[self.hparams.hidden_size, self.hparams.hidden_size],                                           dropout_rate=self.dropout_rate, is_training=self.is_training, scope='construction_group_%s'%domain)

            weigthed_group, attn_weight = self.group_attention(queries=self.normalize(tf.stop_gradient(local_seq), scope='matching'),
                                keys=seq_group,
                                num_units=self.hparams.hidden_size,
                                num_groups=self.num_groups,
                                num_heads=self.num_heads,
                                dropout_rate=self.dropout_rate,
                                is_training=self.is_training,
                                causality=True,
                                #  causality=False,
                                scope=domain)



            weigthed_group = self.feedforward(self.normalize(weigthed_group, scope='matching_%s'%domain), num_units=[self.hparams.hidden_size, self.hparams.hidden_size],                                           dropout_rate=self.dropout_rate, is_training=self.is_training, scope='matching%s'%domain)




            reduced_weigthed_group = tf.reduce_mean(weigthed_group, axis=1)
        return reduced_weigthed_group, attn_weight 
    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        if hparams.test_dropout:
            self.embedding_keep_prob_test = 1.0 - hparams.embedding_dropout
        else:
            self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_outputA, model_outputB, model_output, attn_weight_A, attn_weight_B = self._build_seq_graph()
            # model_output, attn_weight, before_weight, queries, Q, outputs = self._build_seq_graph(domain)

            logitA = self._fcn_net(model_outputA, hparams.layer_sizes, scope="A")
            logitB = self._fcn_net(model_outputB, hparams.layer_sizes, scope="B")

            logit = self._fcn_net(model_output, hparams.layer_sizes, scope="global")

            #self._add_norm()
            return logitA + logit, logitB + logit, attn_weight_A, attn_weight_B

    def trainA(self, sess, feed_dict):
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel, self).trainA(sess, feed_dict)
    def trainB(self, sess, feed_dict):
        # print(feed_dict)

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel, self).trainB(sess, feed_dict)


    #  def batch_train(self, file_iterator, train_sess, vm, tb):
    def batch_trainA(self, file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.
            tb (TensorboardX): visualization manager for TensorboardX.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        epoch_loss = 0
        for batch_data_input in file_iterator:
           # if step == 0:
            #    print("domain A",list(batch_data_input.items())[4])
            if batch_data_input:
                step_result = self.trainA(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, auxiliary_loss, regular_loss, summary) = step_result
                #print(self.step_A, "total", step_loss, "data", step_data_loss)
                #  (_, _, step_loss, step_data_loss, summary, _, _, _, _, _, _) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _,) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, self.step_A)
                epoch_loss += step_loss
                self.step_A += 1
                if self.step_A % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_lossA: {1:.6f}, data_lossA: {2:.6f}, auxiliary_lossA: {3:.6f}, disentangle_lossA: {4:.6f}".format(
                            self.step_A, step_loss, step_data_loss, auxiliary_loss, regular_loss, 
                        )
                    )
                   # print("Gradient A", step_gradient)
                    if self.hparams.visual_type == 'epoch':
                        if vm != None:
                            vm.step_update_line('lossA', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('lossA', step_loss, self.step_A)
               # if step % 600 == 0:  break

                if self.hparams.visual_type == 'step':
                    if step % self.hparams.visual_step == 0:
                        if vm != None:
                            vm.step_update_line('lossA', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('lossA', step_loss, self.step_A)

                        # steps validation for visualization
                        valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                        if vm != None:
                            vm.step_update_multi_lines(valid_res)  # TODO
                        for vs in valid_res:
                            #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                            tb.add_scalar(vs.replace('@', '_'), valid_res[vs], self.step_A)


        return epoch_loss
    def batch_trainB(self, file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.
            tb (TensorboardX): visualization manager for TensorboardX.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        epoch_loss = 0
        for batch_data_input in file_iterator:
           # if step == 0:
            #    print("domain B", list(batch_data_input.items())[4])

            if batch_data_input:
                step_result = self.trainB(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, auxiliary_loss, regular_loss, summary) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _, _, _, _) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _,) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, self.step_B)
                epoch_loss += step_loss
                self.step_B += 1
                if self.step_B % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_lossB: {1:.6f}, data_lossB: {2:.6f}, auxiliary_lossB: {3:.6f}, disentangle_lossB: {4:.6f}".format(
                            self.step_B, step_loss, step_data_loss, auxiliary_loss, regular_loss, 
                        )
                    )
                    #print("gradient B", step_gradient)
                    if self.hparams.visual_type == 'epoch':
                        if vm != None:
                            vm.step_update_line('lossB', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('lossB', step_loss, self.step_B)
               # if step % 600 == 0:   break
                if self.hparams.visual_type == 'step':
                    if self.step_B % self.hparams.visual_step == 0:
                        if vm != None:
                            vm.step_update_line('lossB', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('lossB', step_loss, self.step_B)

                        # steps validation for visualization
                        valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                        if vm != None:
                            vm.step_update_multi_lines(valid_res)  # TODO
                        for vs in valid_res:
                            #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                            tb.add_scalar(vs.replace('@', '_'), valid_res[vs], self.step_B)


        return epoch_loss

    def fit(
        self, train_fileA, train_fileB, valid_fileA, valid_fileB, valid_num_ngs, eval_metric="group_auc", vm=None, tb=None, pretrain=False
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        if not self.need_sample and self.train_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for training without sampling needed."
            )
        if valid_num_ngs < 1:
            raise ValueError(
                "Please specify a positive integer of negative numbers for validation."
            )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        #  if pretrain:
            #  self.saver_emb = tf.train.Saver({'item_lookup':'item_embedding', 'user_lookup':'user_embedding'},max_to_keep=self.hparams.epochs)
        if pretrain:
            print('start saving embedding')
            if not os.path.exists(self.hparams.PRETRAIN_DIR):
                os.makedirs(self.hparams.PRETRAIN_DIR)
            #  checkpoint_emb_path = self.saver_emb.save(
                #  sess=train_sess,
                #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
            #  )
            #  graph_def = tf.get_default_graph().as_graph_def()
            var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
            constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, var_list)
            with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print('embedding saved')

        train_sess = self.sess
        eval_info = list()

        best_metric_A, best_metric_B, self.best_epoch_A, self.best_epoch_B = 0, 0, 0, 0
        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iteratorA = self.iterator.load_data_from_file(
                train_fileA,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            #  epoch_loss = self.batch_train(file_iterator, train_sess, vm, tb)
            epoch_lossA = self.batch_trainA(file_iteratorA, train_sess, vm, tb, valid_fileA, valid_num_ngs)
            #epoch_lossA = 0
            file_iteratorB = self.iterator.load_data_from_file(
                train_fileB,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            epoch_lossB = self.batch_trainB(file_iteratorB, train_sess, vm, tb, valid_fileB, valid_num_ngs)
#            epoch_lossB = 0
            if vm != None:
                vm.step_update_line('epoch lossA', epoch_lossA)
                vm.step_update_line('epoch lossB', epoch_lossB)

            #  tf.summary.scalar('epoch loss', epoch_loss)
            tb.add_scalar('epoch_lossA', epoch_lossA, epoch)
            tb.add_scalar('epoch_lossB', epoch_lossB, epoch)

            valid_resA = self.run_weighted_evalA(valid_fileA, valid_num_ngs)
            valid_resB = self.run_weighted_evalB(valid_fileB, valid_num_ngs)

            print(
                "eval valid at epoch {0}: domain A {1}, domain B {2}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_resA.items()
                        ]
                    ),
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_resB.items()
                        ]
                    ),
                )
            )
            if self.hparams.visual_type == 'epoch':
                if vm != None:
                    vm.step_update_multi_lines(valid_resA)  # TODO
                    vm.step_update_multi_lines(valid_resB)  # TODO

                for vs in valid_resA:
                    #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                    tb.add_scalar(vs.replace('@', '_'), valid_resA[vs], epoch)

                for vs in valid_resB:
                    #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                    tb.add_scalar(vs.replace('@', '_'), valid_resB[vs], epoch)
            eval_info.append((epoch, valid_resA))
            eval_info.append((epoch, valid_resB))
            MODEL_DIR_A = os.path.join(self.hparams.MODEL_DIR, "A/")
            MODEL_DIR_B = os.path.join(self.hparams.MODEL_DIR, "B/")
            progress_A, progress_B = False, False
            early_stop = self.hparams.EARLY_STOP
            if valid_resA[eval_metric] > best_metric_A:
                best_metric_A = valid_resA[eval_metric]
                self.best_epoch_A = epoch
                progress_A = True
            else:
                if early_stop > 0 and epoch - self.best_epoch_A >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))

                    if pretrain:
                        if not os.path.exists(self.hparams.PRETRAIN_DIR):
                            os.makedirs(self.hparams.PRETRAIN_DIR)

                        var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
                        constant_graph = tf.graph_util.convert_variables_to_constants(train_sess, train_sess.graph_def, var_list)
                        with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())

                    break

            if self.hparams.save_model and MODEL_DIR_A:
                if not os.path.exists(MODEL_DIR_A):
                    os.makedirs(MODEL_DIR_A)
                if progress_A:
                    checkpoint_path_A = self.saver.save(
                        sess=train_sess,
                        save_path=MODEL_DIR_A + "epoch_" + str(epoch),
                    )
            early_stop = self.hparams.EARLY_STOP
            if valid_resB[eval_metric] > best_metric_B:
                best_metric_B = valid_resB[eval_metric]
                self.best_epoch_B = epoch
                progress_B = True
            else:
                if early_stop > 0 and epoch - self.best_epoch_B >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))

                    if pretrain:
                        if not os.path.exists(self.hparams.PRETRAIN_DIR):
                            os.makedirs(self.hparams.PRETRAIN_DIR)

                        var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
                        constant_graph = tf.graph_util.convert_variables_to_constants(train_sess, train_sess.graph_def, var_list)
                        with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())

                    break

            if self.hparams.save_model and MODEL_DIR_B:
                if not os.path.exists(MODEL_DIR_B):
                    os.makedirs(MODEL_DIR_B)
                if progress_B:
                    checkpoint_path_B = self.saver.save(
                        sess=train_sess,
                        save_path=MODEL_DIR_B + "epoch_" + str(epoch),
                    )

        if self.hparams.write_tfevents:
            self.writer.close()

        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch_A))
        print("best epoch: {0}".format(self.best_epoch_B))
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)


    def run_weighted_evalA(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        cnt = 5
        flag = True
        # last_user = None

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:
                    step_user, step_pred, step_labels, attn_weight = self.eval_with_userA(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res


    def run_weighted_evalB(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        # last_user = None

        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1
        if calc_mean_alpha:
            alphas = []
        cnt = 5 
        flag = True
        
        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha:

                    step_user, step_pred, step_labels, attn_weight  = self.eval_with_userB(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_userA(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.predA, self.iterator.labels, self.attn_weight_A], feed_dict=feed_dict)
    def eval_with_userB(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.predB, self.iterator.labels, self.attn_weight_B], feed_dict=feed_dict)


    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output], feed_dict=feed_dict)

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def infer(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).infer(sess, feed_dict)

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim
        self.num_groups = hparams.num_groups
        with tf.variable_scope("embedding", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            self.user_lookup = tf.get_variable(
                name="user_embedding",
                shape=[self.user_vocab_length, self.user_embedding_dim],
                dtype=tf.float32,
            )
            self.user_group_emb = tf.get_variable(
                name = 'group_embedding', 
                dtype = tf.float32, 
                shape = [self.num_groups, self.user_embedding_dim],
            )

            self.item_lookup = tf.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup_A = tf.get_variable(
                name="item_embedding_A",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.item_lookup_B = tf.get_variable(
                name="item_embedding_B",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup_A = tf.get_variable(
                name="cate_embedding_A",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup_B = tf.get_variable(
                name="cate_embedding_B",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.position = tf.get_variable(
                name="position_embedding",
                shape=[hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.position_A = tf.get_variable(
                name="position_embeddingA",
                shape=[hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim],
                dtype=tf.float32,
            )
            self.position_B = tf.get_variable(
                name="position_embeddingB",
                shape=[hparams.max_seq_length, self.item_embedding_dim + self.cate_embedding_dim],
                dtype=tf.float32,
            )

        print(self.hparams.FINETUNE_DIR)
        print(not self.hparams.FINETUNE_DIR)
        if self.hparams.FINETUNE_DIR:
            with tf.Session() as sess:
                # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
                #     graph_def = tf.GraphDef()
                #     graph_def.ParseFromString(f.read())
                #     sess.graph.as_default()
                #     tf.import_graph_def(graph_def, name='')
                #  tf.global_variables_initializer().run()
                output_graph_def = tf.GraphDef()
                with open(self.hparams.FINETUNE_DIR + "test-model.pb", "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")

                self.item_lookup = sess.graph.get_tensor_by_name('sequential/embedding/item_embedding')
                self.user_lookup = sess.graph.get_tensor_by_name('sequential/embedding/user_embedding')
            #  print(input_x.eval())

            #  output = sess.graph.get_tensor_by_name("conv/b:0")

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        self.user_embedding = tf.nn.embedding_lookup( # iterator_users overlap?
            self.user_lookup, self.iterator.users
        )
        # tf.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        tf.summary.histogram(
                "item_embedding", self.item_embedding
        )
        self.position_embedding = tf.nn.embedding_lookup(
            self.position, tf.tile(tf.expand_dims(tf.range(tf.shape(self.iterator.item_history)[1]), 0), [tf.shape(self.iterator.item_history)[0], 1]), # B*L 
        )
        self.item_embedding_A = tf.nn.embedding_lookup(
            self.item_lookup_A, self.iterator.items
        )
        tf.summary.histogram(
            "item_embedding_A_output", self.item_embedding_A
        )

        self.item_history_embedding_A = tf.nn.embedding_lookup(
            self.item_lookup_A, self.iterator.item_history
        )
        self.position_embedding_A = tf.nn.embedding_lookup(
            self.position_A, tf.tile(tf.expand_dims(tf.range(tf.shape(self.iterator.item_history)[1]), 0), [tf.shape(self.iterator.item_history)[0], 1]), # B*L 
        )
        # tf.summary.histogram(
        #     "position_embedding_outputA", self.position_embedding_A
        # )
        self.cate_embedding_A = tf.nn.embedding_lookup(
            self.cate_lookup_A, self.iterator.cates
        )
        self.cate_history_embedding_A = tf.nn.embedding_lookup(
            self.cate_lookup_A, self.iterator.item_cate_history
        )
        self.target_item_embedding_A = tf.concat(
            [self.item_embedding_A, self.cate_embedding_A], -1
        )
        self.position_embedding_A = self._dropout(
            self.position_embedding_A, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding_A = self._dropout(
            self.item_history_embedding_A, keep_prob=self.embedding_keeps
        )

        self.cate_history_embedding_A = self._dropout(
            self.cate_history_embedding_A, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding_A = self._dropout(
            self.target_item_embedding_A, keep_prob=self.embedding_keeps
        )


        self.item_embedding_B = tf.nn.embedding_lookup(
            self.item_lookup_B, self.iterator.items
        )
        tf.summary.histogram(
            "item_embedding_B_output", self.item_embedding_B
        )
        print("heihei, I have summary B")
        self.item_history_embedding_B = tf.nn.embedding_lookup(
            self.item_lookup_B, self.iterator.item_history
        )
        self.position_embedding_B = tf.nn.embedding_lookup(
            self.position_B, tf.tile(tf.expand_dims(tf.range(tf.shape(self.iterator.item_history)[1]), 0), [tf.shape(self.iterator.item_history)[0], 1]), # B*L 
        )
        
        self.cate_embedding_B = tf.nn.embedding_lookup(
            self.cate_lookup_B, self.iterator.cates
        )
        self.cate_history_embedding_B = tf.nn.embedding_lookup(
            self.cate_lookup_B, self.iterator.item_cate_history
        )
        self.target_item_embedding_B = tf.concat(
            [self.item_embedding_B, self.cate_embedding_B], -1
        )
        self.position_embedding_B = self._dropout(
            self.position_embedding_B, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding_B = self._dropout(
            self.item_history_embedding_B, keep_prob=self.embedding_keeps
        )

        self.cate_history_embedding_B = self._dropout(
            self.cate_history_embedding_B, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding_B = self._dropout(
            self.target_item_embedding_B, keep_prob=self.embedding_keeps
        )

        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_history
        )

        tf.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.cates
        )
        self.cate_history_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_history
        )
        tf.summary.histogram(
            "cate_history_embedding_output", self.cate_history_embedding
        )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        self.target_item_embedding = tf.concat(
            [self.item_embedding, self.cate_embedding], -1
        )
        tf.summary.histogram("target_item_embedding_output", self.target_item_embedding)

        # dropout after embedding
        self.user_embedding = self._dropout(
            self.user_embedding, keep_prob=self.embedding_keeps
        )
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps
        )
        self.position_embedding = self._dropout(
            self.position_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_history_embedding = self._dropout(
            self.cate_history_embedding, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)

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
        # print("i am fcn in seq base model")
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


    def multihead_attention(self, queries, 
                            keys, 
                            num_units=None, 
                            num_heads=8, 
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention", 
                            reuse=None,
                            with_qk=False):
        '''Applies multihead attention.
        
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked. 
          因果关系：布尔值。 如果为true，则屏蔽引用未来的单位。
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
          A 3d tensor with shape of (N, T_q, C)  
        '''

        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]
            
            # Linear projections
            # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
            # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
            if (scope.split('_')[0]=='groupConstruction'):
                queries = tf.tile(tf.expand_dims(queries, axis=0), [tf.shape(keys)[0], 1, 1])
                Q = queries
            else:
                Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)

            K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
            
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            # Key Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            
            # 和下面query mask的区别：mask值不是设为0，而是设置为无穷小负值（原因是下一步要进行softmax，如果if不执行）
            paddings = tf.ones_like(outputs)*(-2**32+1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Causality = Future blinding
            if causality:
                # 构建下三角为1的tensor
                diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
       
                paddings = tf.ones_like(masks)*(-2**32+1)
                # 下三角置为无穷小负值（原因是下一步要进行softmax）
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
      
            # Activation
            outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
             
            # Query Masking
            # tf.sign输出-1,0,1
            # 根据绝对值之和的符号判定是否mask，效果：某个sequence的特征全为0时（之前被mask过了），mask值为0，否则为1
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1)) # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
            outputs *= query_masks # broadcasting. (N, T_q, C)
              
            # Dropouts
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
                   
            # Weighted sum
            outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
            
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                  
            # Residual connection
            outputs += queries
                  
            # Normalize
            #outputs = normalize(outputs) # (N, T_q, C)
     
        if with_qk: return Q,K
        else: return outputs
    def group_attention(self, queries, 
                            keys, 
                            num_units=None, 
                            num_groups=None, 
                            num_heads=8, 
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention", 
                            reuse=None,
                            with_qk=False):
        '''Applies multihead attention.
        
        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked. 
          因果关系：布尔值。 如果为true，则屏蔽引用未来的单位。
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns
          A 3d tensor with shape of (N, T_q, C)  
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]
            
            # Linear projections

            queries = tf.transpose(tf.layers.dense(tf.transpose(queries, [0,2,1]), num_groups, activation=None), [0,2,1]) # (N, T_q, C)

            Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_k, C)
          #  K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        #    return Q * K, Q, K, queries, keys,Q

         #   return Q * K
            V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
            

            Q_ = Q
           # K_ = K
            V_ = V

            outputs = Q_
            layer_sizes = [ 1]
            activations = ['relu']
            with tf.variable_scope("group_attn"):
                last_layer_size = outputs.shape[-1]
                layer_idx = 0
                hidden_nn_layers = []
                hidden_nn_layers.append(outputs)
                with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                    for idx, layer_size in enumerate(layer_sizes):
                        curr_w_nn_layer = tf.get_variable(
                            name="w_nn_layer" + str(layer_idx),
                            shape=[last_layer_size, layer_size],
                            dtype=tf.float32,
                        )
                        curr_b_nn_layer = tf.get_variable(
                            name="b_nn_layer" + str(layer_idx),
                            shape=[layer_size],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                        )
                        tf.summary.histogram(
                            "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                        )
                        tf.summary.histogram(
                            "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                        )
                        curr_hidden_nn_layer = (
                            tf.tensordot(
                                hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                            )
                            + curr_b_nn_layer
                        )

                        scope = "nn_part" + str(idx)
                        activation = activations[idx]
                        last_beforeNorm = curr_hidden_nn_layer # no problem

                        last_beforeActive = curr_hidden_nn_layer

                        last_afterActive = curr_hidden_nn_layer
                        hidden_nn_layers.append(curr_hidden_nn_layer)
                        layer_idx += 1
                        last_layer_size = layer_size


                    nn_output = hidden_nn_layers[-1]

            attn_weight = tf.nn.softmax(nn_output, axis=1) # (h*N, T_q, T_k)
            tf.summary.histogram(
                            "group_weight_for_each_use", attn_weight
                        )


            outputs = tf.multiply(attn_weight, V_) # ( h*N, T_q, C/h)
            

        if with_qk: return Q,K
        else: return outputs, attn_weight

    def feedforward(self, inputs, 
                    num_units=[2048, 512],
                    scope="multihead_attention", 
                    dropout_rate=0.2,
                    is_training=True,
                    reuse=None):
        '''Point-wise feed forward net.
        
        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            #  outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            
            # Residual connection
            outputs += inputs
            
            # Normalize
            #outputs = normalize(outputs)
        
        return outputs