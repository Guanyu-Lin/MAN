# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os.path import join
import abc
import time
import os
import socket
import numpy as np
import tensorflow as tf
from tensorflow import keras
#  from tensorflow.python import debug as tfdbg
#  np.set_printoptions(threshold=np.inf)
from reco_utils.recommender.deeprec.deeprec_utils import cal_metric, dice


__all__ = ["BaseModel"]


class BaseModel:
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.graph = graph if graph is not None else tf.Graph()
        self.iterator = iterator_creator(hparams, self.graph)
        self.train_num_ngs = (
            hparams.train_num_ngs if "train_num_ngs" in hparams else None
        )

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name="layer_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.placeholder(
                tf.bool, shape=(), name="is_training"
            )
            self.group = tf.placeholder(tf.int32, shape=(), name="group")

            self.initializer = self._get_initializer()

            self.logitA, self.logitB, self.attn_weight_A, self.attn_weight_B= self._build_graph()
            
            self.predA = self._get_pred(self.logitA, self.hparams.method)
            self.lossA = self._get_lossA()
            self.train_step_A = self._train_opt("A")
            self.updateA = self._build_train_optA()
            self.mergedA = self._add_summariesA()
            self.train_step_B = self._train_opt("B")
            self.predB = self._get_pred(self.logitB, self.hparams.method)

            self.lossB = self._get_lossB()


            self.updateB = self._build_train_optB()
            self.mergedB = self._add_summariesB()
            self.saver = tf.train.Saver(max_to_keep=self.hparams.epochs)

            self.extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS
            )
            self.init_op = tf.global_variables_initializer()


        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )
        #  self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(self.init_op)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _get_lossA(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_lossA = self._compute_data_lossA()
        self.disentangle_lossA =tf.reduce_sum( self._disentangle_loss())
        
        self.lossA = tf.add(self.data_lossA, self.disentangle_lossA)
        return self.lossA

    def _get_lossB(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
#        self.data_lossB = tf.add(self._compute_data_lossB(), self._disentangle_loss())
        self.data_lossB = self._compute_data_lossB()
        self.disentangle_lossB = tf.reduce_sum(self._disentangle_loss())
#        self.lossB = self.data_lossB
        self.lossB = tf.add(self.data_lossB, self.disentangle_lossB)
        return self.lossB


#        return self.data_lossB

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        
        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)
        
        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def _add_summariesA(self):
        # tf.summary.scalar("lossA", self.data_lossA)
        # merged = tf.summary.merge_all()
        merged = tf.summary.merge([tf.summary.scalar("lossA", self.data_lossA)])

        return merged

    def _add_summariesB(self):
       
        merged = tf.summary.merge([tf.summary.scalar("lossB", self.data_lossB)])
        return merged

    def _disentangle_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if not(i < j):
                    l2_loss = tf.add(
                        l2_loss, tf.multiply(-0.00001, tf.nn.l2_loss(tf.nn.embedding_lookup(self.user_group_emb, [i]) - tf.nn.embedding_lookup(self.user_group_emb, [j])))
                    )


        return l2_loss


    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.embed_l1, tf.norm(param, ord=1))
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            obj: Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _get_initializer(self):
        if self.hparams.init_method == "tnormal":
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "uniform":
            return tf.random_uniform_initializer(
                -self.hparams.init_value, self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "normal":
            return tf.random_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "xavier_normal":
            return tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        elif self.hparams.init_method == "xavier_uniform":
            return tf.contrib.layers.xavier_initializer(uniform=True, seed=self.seed)
        elif self.hparams.init_method == "he_normal":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=False, seed=self.seed
            )
        elif self.hparams.init_method == "he_uniform":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=True, seed=self.seed
            )
        else:
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )

    def _compute_data_lossA(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logitA, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.reshape(self.predA, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.losses.log_loss(
                    predictions=tf.reshape(self.predA, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logitA, (-1, group))
            if self.hparams.model_type == "NextItNet":
                labels = (
                    tf.transpose(
                        tf.reshape(
                            self.iterator.labels,
                            (-1, group, self.hparams.max_seq_length),
                        ),
                        [0, 2, 1],
                    ),
                )
                labels = tf.reshape(labels, (-1, group))
            else:
                labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss
    def _compute_data_lossB(self):
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logitB, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.reshape(self.predB, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.losses.log_loss(
                    predictions=tf.reshape(self.predB, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logitB, (-1, group))
            if self.hparams.model_type == "NextItNet":
                labels = (
                    tf.transpose(
                        tf.reshape(
                            self.iterator.labels,
                            (-1, group, self.hparams.max_seq_length),
                        ),
                        [0, 2, 1],
                    ),
                )
                labels = tf.reshape(labels, (-1, group))
            else:
                labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss
    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            obj: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)

    def _train_opt(self, domain):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "adam":
            train_step = tf.train.AdamOptimizer(lr, name=domain)
        elif optimizer == "ftrl":
            train_step = tf.train.FtrlOptimizer(lr)
        elif optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "padagrad":
            train_step = tf.train.ProximalAdagradOptimizer(lr)
        elif optimizer == "pgd":
            train_step = tf.train.ProximalGradientDescentOptimizer(lr)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "lazyadam":
            train_step = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr)
        return train_step

    def _build_train_optA(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        # with tf.variable_scope("domain_A"):
        
        gradients, variables = zip(*self.train_step_A.compute_gradients(self.lossA))
        
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return self.train_step_A.apply_gradients(zip(gradients, variables))

    def _build_train_optB(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """

        # train_step = self._train_opt("B")
        gradients, variables = zip(*self.train_step_B.compute_gradients(self.lossB))
        
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return self.train_step_B.apply_gradients(zip(gradients, variables))
       
    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.
        
        Args:
            logit (obj): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters
        
        Returns:
            obj: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation, layer_idx)

    def _activate(self, logit, activation, layer_idx=-1):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        elif activation == 'dice':
            return dice(logit, name='dice_{}'.format(layer_idx))
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.
        Args:
            logit (obj): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            obj: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

    def trainA(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        

        return sess.run(
            [
                self.updateA,
                self.extra_update_ops,
                self.lossA,
                self.data_lossA,
                self.disentangle_lossA,
                self.disentangle_lossA,
                self.mergedA,
                        
            ],
            feed_dict=feed_dict,
        )

    def trainB(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        # import pdb
        # pdb.set_trace()
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.updateB,
                self.extra_update_ops,
                self.lossB,
                self.data_lossB,
                self.disentangle_lossB,
                self.disentangle_lossB,
                self.mergedB,
                        #  self.pp1,
                #  self.pp2,
                #  self.pp3,
                #  self.pp4,
                #  self.pp5,
                #  self.pp6,
            ],
            feed_dict=feed_dict,
        )
    def eval(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict)

    def infer(self, sess, feed_dict):
        """Given feature data (in feed_dict), get predicted scores with current model.
        Args:
            sess (obj): The model session object.
            feed_dict (dict): Instances to predict. This is a dictionary that maps graph elements to values.

        Returns:
            list: Predicted scores for the given instances.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred], feed_dict=feed_dict)

    def load_model(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: model path.

        Raises:
            IOError: if the restore operation failed.
        """
        act_path = self.hparams.load_saved_model
        if model_path is not None:
            act_path = model_path

        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))

    def fit(self, train_file, valid_file, test_file=None):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_file (str): test set.

        Returns:
            obj: An instance of self.
        """
        if self.hparams.write_tfevents:
            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            for (
                batch_data_input,
                impression,
                data_size,
            ) in self.iterator.load_data_from_file(train_file):
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            if self.hparams.save_model:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if epoch % self.hparams.save_epoch == 0:
                    save_path_str = join(self.hparams.MODEL_DIR, "epoch_" + str(epoch))
                    checkpoint_path = self.saver.save(
                        sess=train_sess, save_path=save_path_str
                    )

            eval_start = time.time()
            eval_res = self.run_eval(valid_file)
            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        if self.hparams.write_tfevents:
            self.writer.close()

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.
        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.
        Returns:
            all_labels: labels after group.
            all_preds: preds after group.
        """
        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        return all_labels, all_preds

    def run_eval(self, filename):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        load_sess = self.sess
        preds = []
        labels = []
        imp_indexs = []
        for batch_data_input, imp_index, data_size in self.iterator.load_data_from_file(
            filename
        ):
            step_pred, step_labels = self.eval(load_sess, batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexs.extend(np.reshape(imp_index, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        if self.hparams.pairwise_metrics is not None:
            group_labels, group_preds = self.group_labels(labels, preds, imp_indexs)
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
        return res

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name, format is same as train/val/test file.
            outfile_name (str): Output file name, each line is the predict score.

        Returns:
            obj: An instance of self.
        """
        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input, _, data_size in self.iterator.load_data_from_file(
                infile_name
            ):
                step_pred = self.infer(load_sess, batch_data_input)
                step_pred = step_pred[0][:data_size]
                step_pred = np.reshape(step_pred, -1)
                wt.write("\n".join(map(str, step_pred)))
                # line break after each batch.
                wt.write("\n")
        return self

    def _attention(self, inputs, attention_size):
        """Soft alignment attention implement.
        
        Args:
            inputs (obj): Sequences ready to apply attention.
            attention_size (int): The dimension of attention operation.

        Returns:
            obj: Weighted sum after attention.
        """
        hidden_size = inputs.shape[2].value
        if not attention_size:
            attention_size = hidden_size

        attention_mat = tf.get_variable(
            name="attention_mat",
            shape=[inputs.shape[-1].value, hidden_size],
            initializer=self.initializer,
        )
        att_inputs = tf.tensordot(inputs, attention_mat, [[2], [0]])

        query = tf.get_variable(
            name="query",
            shape=[attention_size],
            dtype=tf.float32,
            initializer=self.initializer,
        )
        att_logits = tf.tensordot(att_inputs, query, axes=1, name="att_logits")
        att_weights = tf.nn.softmax(att_logits, name="att_weights")
        output = inputs * tf.expand_dims(att_weights, -1)
        return output

    def _fcn_net(self, model_output, layer_sizes, scope):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
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
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                tf.summary.histogram(
                    "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
                )
                tf.summary.histogram(
                    "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
                )
                nn_output = (
                    tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1)
                    + b_nn_output
                )
                self.logit = nn_output
                return nn_output


    def _ep_fcn_net(self, model_output, layer_sizes, scope, domain):
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope):
            tab_gate_param1 = tf.get_variable("tab_gate_param1", shape=(8, 31))
            tab_gate_param2 = tf.get_variable("tab_gate_param2", shape=(32, 224))
            bias_input = tf.ones((tf.shape(model_output)[0], 1))
            if domain == "A":
                gate_input = self.cate_embedding
                # gate_input = tf.tile(self.cate_embedding, [tf.shape(model_output)[0], 1])
            else:
                gate_input = self.cate_embedding
                # gate_input = tf.tile(self.cate_embedding, [tf.shape(model_output)[0], 1])
            id_embedding = tf.concat([self.user_embedding, self.target_item_embedding], 1)
            tab_gate_out1 = tf.nn.relu(tf.matmul(gate_input, tab_gate_param1))
            # pdb.set_trace()
            tab_gate_out2 = tf.nn.sigmoid(tf.matmul(tf.concat([tab_gate_out1, bias_input], 1), tab_gate_param2))

            tab_gate = 2.0 * tab_gate_out2

            epnet_output = model_output * tab_gate
            combine_only_input = (tf.concat([epnet_output, model_output], 1))
            # pdb.set_trace()
            combine_input = tf.concat([combine_only_input, id_embedding], 1)
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    g1 = self.gate_nn(combine_input, layer_size, last_layer_size, 'gate_%d'%idx)

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
#                    pdb.set_trace()
                    curr_hidden_nn_layer = (
                        tf.tensordot(
                            g1 * hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                        )
                        + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                tf.summary.histogram(
                    "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
                )
                tf.summary.histogram(
                    "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
                )
                nn_output = (
                    tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1)
                    + b_nn_output
                )
                self.logit = nn_output
                return nn_output

    def ppnet(self, epnet_output, embedding_input, id_embedding):
        """
        epnet_output: epnet的输出
        embedding_input: 底层输入的embedding: Sequential embedding
        id_embedding: uid, aid, pid的embedding
        """
        with tf.name_scope('ppnet'):
            combine_only_input = tf.stop_gradient(tf.concat([epnet_output, embedding_input], 1))
            combine_input = tf.concat([combine_only_input, id_embedding], 1)
        
            g1 = gate_nn(combine_input, 512, combine_input.get_shape()[1], 'gate1')
            h1 = tf.layers.dense(g1 * combine_input, 512, activation=tf.nn.relu, name='h1')
        
            g2 = gate_nn(combine_input, 512, 512, 'gate2')
            h2 = tf.layers.dense(g2 * h1, 256, activation=tf.nn.relu, name='h2')
        
            g3 = gate_nn(combine_input, 512, 256, 'gate3')
            h3 = tf.layers.dense(g3 * h2, 128, activation=tf.nn.relu, name='h3')
        
            g4 = gate_nn(combine_input, 512, 128, 'gate4')
            h4 = tf.layers.dense(g4 * h3, 128, activation=tf.nn.relu, name='h4')
            
            ppnet_output = tf.layers.dense(h4, 1, activation=tf.nn.sigmoid, name='h5')
            return ppnet_output

    def gate_nn(self, inputs, unit1, unit2, name):
        """
        inputs: 网络输入
        unit1: dense层1的输出维度
        unit2: dense层2的输出维度
        name: dense层的名字
        """
        with tf.name_scope('{}_lhuc'.format(name)):
            output = inputs
            with tf.name_scope('{}_lhuc_layer_{}'.format(name, 0)):
                output = tf.layers.dense(output, unit1, activation=tf.nn.relu, name='dense_{}_{}'.format(name, 0))
            with tf.name_scope('{}_lhuc_layer_{}'.format(name, 1)):
                output = 2.0 * tf.layers.dense(output, unit2, activation=tf.nn.sigmoid, name='dense_{}_{}'.format(name, 1))
            return output

    def epnet(self, embedding_input, domain):
        """
        embedding_input: 底层输入的embedding, sequential embedding
        tab_id：三大tab的id embedding, domain 
        bias_fea: 用户侧的bias特征，如是否为新用户，是否为低活用户等 
        id_embedding: uid, aid, pid的embedding
        """
        
        with tf.name_scope('embedding_gate'):
            tab_gate_param1 = tf.get_variable("tab_gate_param1", shape=(72, 511))
            tab_gate_param2 = tf.get_variable("tab_gate_param2", shape=(512, 2376))
            bias_input = tf.ones((tf.shape(embedding_input)[0], 1))
            if domain == "A":
                gate_input = self.domain_A_emb
            else:
                gate_input = self.domain_B_emb
            id_embedding = tf.concat([self.user_embedding, self.target_item_embedding], 2)
            tab_gate_out1 = tf.nn.relu(tf.matmul(gate_input, tab_gate_param1))
            tab_gate_out2 = tf.nn.sigmoid(tf.matmul(tf.concat([tab_gate_out1, bias_input], 1), tab_gate_param2))
            tab_gate = 2.0 * tab_gate_out2
            epnet_output = embedding_input * tab_gate
            ppnet_output = ppnet(epnet_output, embedding_input, id_embedding)
            return ppnet_output
