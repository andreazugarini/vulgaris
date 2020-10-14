"""
Configuration file. Here all the options and hyperparameters are setup within a Config object. 
There are several classes to create configurations for different parts of a model. 
In the end of the file, the setup_config() function is used to initialize the configs of an experiment.
"""
import tensorflow as tf


class Config(object):
    def __init__(self):
        """
        Configuration Class
        """

    def set_params(self, dict):
        def _recursive_call(items, attr_id):
            items = list(items)
            attr_name = items[attr_id][0]
            attr_values = items[attr_id][1]
            for attr_value in attr_values:
                setattr(self, attr_name, attr_value)
                if attr_id == (len(items) - 1):  # base case[
                    yield self
                else:
                    for i in _recursive_call(items, attr_id + 1):
                        yield i

        items = dict.items()
        for i in _recursive_call(items, 0):
            yield i

    def wrap_config(self, properties_dict):
        for k,v in properties_dict.items():
            setattr(self, k, v)
        return self


class BasicLanguageConfig(Config):
    def __init__(self, vocab_size=1880, input_emb_size=300, input_seq_max_len=50, output_seq_max_len=10,
                 input_vocab_size=None, output_vocab_size=None):
        super(Config, self).__init__()

        self.input_vocab_size = input_vocab_size if input_vocab_size else vocab_size
        self.output_vocab_size = output_vocab_size if output_vocab_size else vocab_size
        self.input_emb_size = input_emb_size
        self.output_emb_size = self.input_emb_size
        self.seq_max_len = input_seq_max_len
        self.output_seq_max_len = output_seq_max_len

        self.set_special_tokens()

    def set_special_tokens(self):
        vocab_size = self.input_vocab_size
        self.special_tokens = []
        self._PAD = vocab_size - 2
        self.special_tokens.append(('<PAD>', self._PAD))
        self._GO = vocab_size - 4
        self.special_tokens.append(('<GO>', self._GO))
        self._EOS = vocab_size - 1
        self.special_tokens.append(('<EOS>', self._EOS))
        self._EOT = vocab_size - 3  # end of terzina
        self.special_tokens.append(('<EOT>', self._EOT))
        self._SEP = vocab_size - 5
        self.special_tokens.append(('<SEP>', self._SEP))
        self._UNK = 0

    def get_special_tokens_ids(self):
        return [t[1] for t in self.special_tokens]

    def set_tied_params(self):
        self.output_emb_size = self.input_emb_size


class LearningConfig(Config):
    def __init__(self, optimizer=tf.compat.v1.train.AdagradOptimizer, lr=0.001, norm_clip=100.0, batch_size=32, max_no_improve=5, n_epochs=10,
                 trunc_norm_init_std=1e-4, is_test=False):
        super(Config, self).__init__()

        self.batch_size = batch_size
        self.n_steps = 3500
        self.n_epochs = n_epochs
        self.lr = lr
        # self.optimizer = tf.train.AdamOptimizer
        self.optimizer = optimizer
        self.norm_clip = norm_clip

        self.max_no_improve = max_no_improve

        self.trunc_norm_init_std = trunc_norm_init_std

        self.is_test = is_test


class EncoderConfig(Config):
    def __init__(self, rnn_size, cell_type="LSTM", keep_prob=0.7, mask_pad=True,  is_test=False):
        super(Config, self).__init__()

        # Encoder
        self.rnn_size = rnn_size
        self.is_test = is_test
        self.keep_prob = keep_prob if not self.is_test else 1.0
        self.proj_layers = None
        self.cell_type = cell_type
        self.mask_pad = mask_pad

        self.use_pretrained_embeddings = False
        self.embeddings = None
        self.rand_unif_init = 0.02
        self.use_seq_len = False

    def setIsTest(self, is_test):
        self.is_test = is_test
        self.keep_prob = self.keep_prob if not is_test else 1.0

    def setEmbeddings(self, use_pretrained_embeddings, embeddings=None):
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embeddings = embeddings

    def isLSTM(self):
        return self.cell_type == "LSTM"


class LMConfigs(Config):
    def __init__(self, language, encoder, learning=None,
                 is_test=False, is_generation=False, restore_model=False):
        super(Config, self).__init__()
        self.language = language
        self.encoder = encoder
        self.learning = learning

        self.is_test = is_test
        self.is_generation = is_generation
        self.restore_model = restore_model
        self.restore_path = None

    def setLanguageConfig(self, language_config):
        self.language = language_config

