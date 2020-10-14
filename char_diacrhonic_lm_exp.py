"""
File to run Character Language Modeling Experiments on Middle-Age data.
It computes PLD and PLR metrics on four varieties of the corpus:
    - 1200-1300 (XIII)
    - 1300-1400 (XIV)
    - 1450-1550 (XV-XVI-1)
    - 1500-1600 (XV-XVI-2)

How to run it:
    python char_diachronic_lm_exp.py path/to/vulgaris.csv
"""

import os, sys
import tensorflow as tf

from datasets.char_lm_dataset import CharLMDataset
from models.language_model import LanguageModel
from experiments.lm_experiment import LMExperiment
from utils.basic_utils import save_data, load_data
from distances.pld import pld
from distances.plr import plr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CHOOSE GPU HERE

FLAGS = tf.app.flags.FLAGS

# Vocabulary and Embeddings
tf.app.flags.DEFINE_integer('vocab_size', 130, 'size of the vocabulary')
tf.app.flags.DEFINE_integer('emb_size', 50, 'size of word embeddings')
tf.app.flags.DEFINE_bool('reuse_wes', True, 'share word embeddings in input with the one in output')

# Author/Family Embs
tf.app.flags.DEFINE_bool('use_author_embs', True, 'use author embs or not')
tf.app.flags.DEFINE_bool('use_family_embs', True, 'use family embs or not')
tf.app.flags.DEFINE_bool('use_is_prose_embs', True, 'use pros embs or not')

tf.app.flags.DEFINE_integer('author_vocab_size', 104, 'size of the authors vocabulary')
tf.app.flags.DEFINE_integer('author_emb_size', 16, 'size of authors embeddings')

tf.app.flags.DEFINE_integer('family_vocab_size', 15, 'size of the family vocabulary')
tf.app.flags.DEFINE_integer('family_emb_size', 16, 'size of family embeddings')

tf.app.flags.DEFINE_integer('is_prose_emb_size', 32, 'size of family embeddings')

# RNNs
tf.app.flags.DEFINE_string('rnns_cell_type', "LSTM", 'cell type')

# Encoder
tf.app.flags.DEFINE_integer('input_seq_max_len', 50, 'input sequence max length ')
tf.app.flags.DEFINE_integer('enc_size', 256, 'size of encoder state (if bidirectional it is doubled) ')
tf.app.flags.DEFINE_float('enc_keep_prob', 0.8, 'encoder dropout probability')
tf.app.flags.DEFINE_integer('output_seq_max_len', 50, 'output sequence max length ')

# Learning
tf.app.flags.DEFINE_bool('restore_model', False, 'restore weights from a pretrained model')
tf.app.flags.DEFINE_integer('n_epochs', 30, 'max number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('norm_clip', 1.0, 'gradient clip norm')
tf.app.flags.DEFINE_integer('max_no_improve', 3, 'number of validations tests without improvements')



if __name__ == '__main__':
    filename = sys.argv[1] #

    # define varieties
    # choose here how to group families together
    varieties = {
        "1200-1300": [
            'Poesia Didattica Del Nord', 'Testi Arcaici', 'Poesia “realistica” Toscana',
            'Poesia Popolare E Giullaresca', 'Laude', 'Dolce Stil Novo', 'Poesia Cortese Toscana E Settentrionale',
            "Poesia Didattica Dell'italia Centrale", 'Vicini Degli Stilnovisti', 'Scuola Siciliana'
        ],
        "1300-1400": ['Boccaccio', 'Petrarca', 'others'],
        "1450-1550": ['Ariosto'],
        "1500-1600": ['Tasso']
    }

    # Create Common Vocabularies, they are the same on all the experiments, otherwise PPL becomes meaningless
    vocabs_path = os.path.join(os.getcwd(), "savings", "char_vocabs")
    if not os.path.exists(vocabs_path):
        os.mkdir(vocabs_path)

    chars_vocab_path = os.path.join(vocabs_path, "char_vocabulary.pkl")
    authors_vocab_path = os.path.join(vocabs_path, "authors_vocabulary.pkl")
    families_vocab_path = os.path.join(vocabs_path, "families_vocabulary.pkl")

    home_path = os.path.join(os.getcwd())
    base_dir = os.path.join(home_path, "savings")  # experiments directory
    exp_dir = "vulgaris_analyse_language_varieties"
    exp_path = os.path.join(base_dir, exp_dir)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)  # this experiment directory

    if not os.path.exists(chars_vocab_path):
        config, _, _ = LanguageModel.get_params(FLAGS)
        dataset = CharLMDataset(config)
        dataset.build(filename, split_size=2)  # to create vocabs

        chars_vocab = dataset.vocabulary
        authors_vocab = dataset.authors_vocabulary
        families_vocab = dataset.families_vocabulary

        save_data(chars_vocab, chars_vocab_path)
        save_data(authors_vocab, authors_vocab_path)
        save_data(families_vocab, families_vocab_path)

    else:
        chars_vocab = load_data(chars_vocab_path)
        authors_vocab = load_data(authors_vocab_path)
        families_vocab = load_data(families_vocab_path)

    diachronic_ppl = {}
    for train_family_name, train_family_group in varieties.items():
        tf.reset_default_graph()
        # for all the varieties, train an lm and measure ppl in the other varieties (itself included)

        tmp_path = os.path.join(exp_path, train_family_name)
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)  # this experiment directory

        # Get Configurations
        config, val_config, _ = LanguageModel.get_params(FLAGS)

        # Loading the Vocabulary and creating the Dataset
        lm_dataset = CharLMDataset(
            config,
            vocabulary=chars_vocab,
            authors_vocabulary=authors_vocab,
            families_vocabulary=families_vocab,
            name=train_family_name
        )

        lm_dataset.build(filename, families=train_family_group, split_size=0.9)

        # Building Model
        x_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])  # batch_size x sentence_max_len
        lm = LanguageModel(x_ph, config, "NMT")
        lm_inference = LanguageModel(x_ph, val_config, "NMT", reuse=True)

        # Experiment Run
        lm_exp = LMExperiment(lm, lm_inference, lm_inference, lm_dataset, tmp_path, config)
        lm_exp.run()

        # set RESTORE PATH AND RESTORE MODEL TO TRUE
        config.restore_model = True
        config.restore_path = tmp_path
        authors_vocab = load_data(os.path.join(vocabs_path, "authors_vocabulary.pkl"))
        families_vocab = load_data(os.path.join(vocabs_path, "families_vocabulary.pkl"))
        for test_family_name,test_family_group in varieties.items():
            # test on all the family varieties
            family_group_dataset = CharLMDataset(
                config,
                vocabulary=chars_vocab,
                authors_vocabulary=authors_vocab,
                families_vocabulary=families_vocab,
                name=test_family_name
            )

            family_group_dataset.build(filename=filename, split_size=-1, families=test_family_group)
            _ppl = lm_exp.test(dataset=family_group_dataset)
            print("==========================================================")
            print(f"Evaluation on {test_family_name} of model trained on {train_family_name}")
            if train_family_name not in diachronic_ppl:
                diachronic_ppl[train_family_name] = {}
            diachronic_ppl[train_family_name][test_family_name] = _ppl
            print(diachronic_ppl)

    varieties_names = list(varieties.keys())

    print("\n\n==========================================================")
    print("==========================================================")
    varieties_pld = pld(diachronic_ppl)
    print(f"PLD: {varieties_pld}\n")

    varieties_plr = plr(diachronic_ppl)
    print(f"PLR: {varieties_plr}")