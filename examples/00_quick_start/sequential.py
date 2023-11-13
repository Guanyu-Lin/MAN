from absl import app
from absl import flags
from absl import logging

import sys
sys.path.append("../../")
import os
import socket
import getpass
import smtplib
from email.mime.text import MIMEText
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import setproctitle

import tensorflow as tf
import time

from reco_utils.common.constants import SEED
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams
)
from reco_utils.dataset.sequential_reviews import data_preprocessing
from reco_utils.dataset.sequential_reviews import group_sequence

from reco_utils.recommender.deeprec.models.sequential.surge import SURGEModel
from reco_utils.recommender.deeprec.models.sequential.sasrec import SASRecModel

from reco_utils.recommender.deeprec.io.sequential_iterator import (
    SequentialIterator,
    SASequentialIterator,
    RecentSASequentialIterator,
    ShuffleSASequentialIterator
)

from reco_utils.common.visdom_utils import VizManager
from tensorboardX import SummaryWriter
from tensorflow.python.tools import inspect_checkpoint as chkp
# import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'Amazon-MAN', 'Experiment name.')

flags.DEFINE_string('datasetA', 'games', 'Dataset name.')
flags.DEFINE_string('datasetB', 'toys', 'Dataset name.')

flags.DEFINE_integer('val_num_ngs', 9, 'Number of negative instances with a positiver instance for validation.')
flags.DEFINE_integer('test_num_ngs', 9, 'Number of negative instances with a positive instance for testing.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.')

flags.DEFINE_string('model', 'MAN', 'Model name.')
flags.DEFINE_string('backbone', 'SURGE', 'Backbone name.')

flags.DEFINE_float('embed_l2', 1e-2, 'L2 regulation for embeddings.')
flags.DEFINE_float('layer_l2', 1e-2, 'L2 regulation for layers.')
flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
flags.DEFINE_integer('contrastive_length_threshold', 10, 'Minimum sequence length value to apply contrastive loss.')
flags.DEFINE_integer('contrastive_recent_k', 5, 'Use the most recent k embeddings to compute short-term proxy.')


flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_boolean('test_dropout', False, 'Whether to dropout during evaluation.')
flags.DEFINE_boolean('write_prediction_to_file', False, 'Whether to write prediction to file.')

flags.DEFINE_integer('counterfactual_recent_k', 10, 'Use recent k interactions to predict the target item.')
flags.DEFINE_boolean('pretrain', False, 'Whether to use pretrain and finetune.')

flags.DEFINE_string('finetune_path', '', 'Save path.')
flags.DEFINE_boolean('vector_alpha', False, 'Whether to use vector alpha for long short term fusion.')
flags.DEFINE_boolean('manual_alpha', False, 'Whether to use predefined alpha for long short term fusion.')
flags.DEFINE_float('manual_alpha_value', 0.5, 'Predifined alpha value for long short term fusion.')
flags.DEFINE_boolean('interest_evolve', True, 'Whether to use a GRU to model interest evolution.')
flags.DEFINE_boolean('predict_long_short', True, 'Predict whether the next interaction is driven by long-term interest or short-term interest.')
flags.DEFINE_enum('single_part', 'no', ['no', 'long', 'short'], 'Whether to use only long, only short or both.')
flags.DEFINE_integer('is_clip_norm', 1, 'Whether to clip gradient norm.')
flags.DEFINE_boolean('use_complex_attention', True, 'Whether to use complex attention like DIN.')
flags.DEFINE_boolean('is_preprocessing', False, 'Whether to preprocess the dataset.')

flags.DEFINE_boolean('use_time4lstm', True, 'Whether to use Time4LSTMCell proposed by SLIREC.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('early_stop', 2, 'Patience for early stop.')
flags.DEFINE_integer('pretrain_epochs', 10, 'Number of pretrain epochs.')
flags.DEFINE_integer('finetune_epochs', 100, 'Number of finetune epochs.')
flags.DEFINE_string('data_path', os.path.join("..",".."), 'Data file path.')

flags.DEFINE_string('save_path', '../../saves/', 'Save path.')
    
flags.DEFINE_integer('train_num_ngs', 9, 'Number of negative instances with a positive instance for training.')
flags.DEFINE_float('sample_rate', 1.0, 'Fraction of samples for training and testing.')
flags.DEFINE_float('attn_loss_weight', 0.001, 'Loss weight for supervised attention.')
flags.DEFINE_float('discrepancy_loss_weight', 0.01, 'Loss weight for discrepancy between long and short term user embedding.')
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('show_step', 100, 'Step for showing metrics.')
flags.DEFINE_integer('max_seq_length', 20, 'Step for showing metrics.')
flags.DEFINE_integer('num_groups', 10, 'Step for showing metrics.')

flags.DEFINE_string('visual_type', 'epoch', '') #  for epoch visual
flags.DEFINE_integer('visual_step', 50, 'Step for drawing metrics.')
flags.DEFINE_boolean('enable_mail_service', False, 'Whether to e-mail yourself after each run.')


def get_model(flags_obj, model_path, summary_path, pretrain_path, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path):

    EPOCHS = flags_obj.epochs
    BATCH_SIZE = flags_obj.batch_size
    RANDOM_SEED = None  # Set None for non-deterministic result

    pairwise_metrics = ['mean_mrr', 'ndcg@2;4;6;8;10']
    weighted_metrics = ['wauc']
    max_seq_length = flags_obj.max_seq_length
    time_unit = 's'

    input_creator = SequentialIterator


    # SURGE
    if flags_obj.backbone == 'SURGE':
        yaml_file = '../../reco_utils/recommender/deeprec/config/gcn.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                num_groups=flags_obj.num_groups, 
                                hidden_size=20,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SURGEModel(hparams, input_creator, seed=RANDOM_SEED)

    # SASRec
    elif flags_obj.backbone == 'SASRec':
        yaml_file = '../../reco_utils/recommender/deeprec/config/sasrec.yaml'
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                num_groups=flags_obj.num_groups, 
                                hidden_size=32,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                    )
        model = SASRecModel(hparams, input_creator, seed=RANDOM_SEED)
    
    return model




def main(argv):

    flags_obj = FLAGS

    setproctitle.setproctitle('{}@LGY'.format(flags_obj.name))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags_obj.gpu_id)

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))
    print('start experiment')

    data_path = os.path.join(flags_obj.data_path, "Amazon_data_processed")
    if flags_obj.datasetA == 'games':
        reviews_nameA = 'ratings_Video_Games.csv'
        meta_nameA = ''
    elif flags_obj.datasetA == 'toys':
        reviews_nameA = 'ratings_Toys_and_Games.csv'
        meta_nameA = ''
    

    if flags_obj.datasetB == 'games':
        reviews_nameB = 'ratings_Video_Games.csv'
        meta_nameB = ''
    elif flags_obj.datasetB == 'toys':
        reviews_nameB = 'ratings_Toys_and_Games.csv'
        meta_nameB = ''
    
    reviews_fileA = os.path.join(data_path, reviews_nameA)
    reviews_fileB = os.path.join(data_path, reviews_nameB)

    meta_fileA = os.path.join(data_path, meta_nameA)
    meta_fileB = os.path.join(data_path, meta_nameB)

    train_fileA = os.path.join(data_path, r'NDCG_train_dataA')
    valid_fileA = os.path.join(data_path, r'NDCG_valid_dataA')
    test_fileA = os.path.join(data_path, r'NDCG_test_dataA')
    user_vocab = os.path.join(data_path, r'NDCG_user_vocab_cd.pkl')
    item_vocab = os.path.join(data_path, r'NDCG_item_vocab_cd.pkl')
    cate_vocab = os.path.join(data_path, r'NDCG_category_vocab_cd.pkl')
    output_fileA = os.path.join(data_path, r'NDCG_outputA.txt')


    train_fileB = os.path.join(data_path, r'NDCG_train_dataB')
    valid_fileB = os.path.join(data_path, r'NDCG_valid_dataB')
    test_fileB = os.path.join(data_path, r'NDCG_test_dataB')
    
    output_fileB = os.path.join(data_path, r'NDCG_outputB.txt')

    train_num_ngs = flags_obj.train_num_ngs
    valid_num_ngs = flags_obj.val_num_ngs
    test_num_ngs = flags_obj.test_num_ngs
    sample_rate = flags_obj.sample_rate

    input_files = [reviews_fileA, meta_fileA, train_fileA, valid_fileA, test_fileA, user_vocab, item_vocab, cate_vocab, reviews_fileB, meta_fileB, train_fileB, valid_fileB, test_fileB]

    
    if flags_obj.is_preprocessing:
        data_preprocessing(*input_files, sample_rate=sample_rate, valid_num_ngs=valid_num_ngs, test_num_ngs=test_num_ngs, datasetA=flags_obj.datasetA, datasetB=flags_obj.datasetB)
    if not os.path.exists(test_fileA+'_group1'):
        if flags_obj.datasetA == 'games':
            split_length = [10, 20, 30, 50]
        elif flags_obj.datasetA == 'toys':
            split_length = [10, 20, 30, 50]
        group_sequence(test_file=test_fileA, split_length=split_length)
    if not os.path.exists(test_fileB+'_group1'):
        if flags_obj.datasetB == 'games':
            split_length = [10, 20, 30, 50]
        elif flags_obj.datasetB == 'toys':
            split_length = [10, 20, 30, 50]
        group_sequence(test_file=test_fileB, split_length=split_length)


  

    save_path = os.path.join(flags_obj.save_path, flags_obj.model, flags_obj.name)
    model_path = os.path.join(save_path, "model/")
    summary_path = os.path.join(save_path, "summary/")
    pretrain_path = os.path.join(save_path, "pretrain/")
    finetune_path = flags_obj.finetune_path

    model = get_model(flags_obj, model_path, summary_path, pretrain_path, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path)


    if flags_obj.only_test:

        model_path_A = os.path.join(model_path, "A/")
        ckpt_path_A = tf.train.latest_checkpoint(model_path_A)
        model.load_model(ckpt_path_A)
        res_synA = model.run_weighted_evalA(test_fileA, num_ngs=test_num_ngs)
        model_path_B = os.path.join(model_path, "B/")
        ckpt_path_B = tf.train.latest_checkpoint(model_path_B)
        model.load_model(ckpt_path_B)
        res_synB = model.run_weighted_evalB(test_fileB, num_ngs=test_num_ngs)
        print(flags_obj.name)
        print(res_synA)
        print(res_synB)
        


    
    vm = None
    visual_path = os.path.join(save_path, "metrics/")
    tb = SummaryWriter(log_dir=visual_path, comment='tb')


    eval_metric = 'wauc'
    

    start_time = time.time()
    model = model.fit(train_fileA, train_fileB, valid_fileA, valid_fileB, valid_num_ngs=valid_num_ngs, eval_metric=eval_metric, vm=vm, tb=tb, pretrain=flags_obj.pretrain) 

    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))


    start_time = time.time()
    model_path_A = os.path.join(model_path, "A/")
    ckpt_path_A = tf.train.latest_checkpoint(model_path_A)
    model.load_model(ckpt_path_A)
    res_synA = model.run_weighted_evalA(test_fileA, num_ngs=test_num_ngs)

    model_path_B = os.path.join(model_path, "B/")
    ckpt_path_B = tf.train.latest_checkpoint(model_path_B)
    model.load_model(ckpt_path_B)

    res_synB = model.run_weighted_evalB(test_fileB, num_ngs=test_num_ngs)
    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for testing is {0:.2f} mins'.format((cost_time)/60.0))

    print(flags_obj.name)
    print(res_synA)
    print(res_synB)

    

    if flags_obj.write_prediction_to_file:
        model = model.predict(test_file, output_file)


if __name__ == "__main__":
    
    app.run(main)
