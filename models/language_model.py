import tensorflow as tf
import copy

from configs.configs import BasicLanguageConfig, LearningConfig, EncoderConfig, LMConfigs


def cell(size, cell_type, attention=False, attention_size=None, dropout=None, proj=None, layers=1, initializer=None, config=None):
    cells = []
    _cell = None
    for _ in range(layers):
        if cell_type == "LSTM":
            _cell = tf.contrib.rnn.LSTMCell(size, initializer=initializer)
        elif cell_type == "GRU":
            _cell = tf.contrib.rnn.GRUCell(size)
        else:
            _cell = tf.contrib.rnn.BasicRNNCell(size)

        if isinstance(proj, (list,)):
            # adds multiple projections to the output of the rnn
            for i in range(len(proj)):
                _cell = tf.contrib.rnn.OutputProjectionWrapper(_cell, proj[i])
        elif proj:
            _cell = tf.contrib.rnn.OutputProjectionWrapper(_cell, proj)

        if attention and attention_size:
            is_tuple = True if cell_type == "LSTM" else False
            _cell = tf.contrib.rnn.AttentionCellWrapper(_cell,
                                                        attn_length=attention_size,
                                                        state_is_tuple=is_tuple)

        if dropout:
            _cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                _cell, dtype=tf.float32,
                input_keep_prob=1.0,
                output_keep_prob=dropout,
                state_keep_prob=1.0
            )
        cells.append(_cell)
    if layers == 1:
        return _cell
    else:
        return tf.contrib.rnn.MultiRNNCell(cells)


class SimpleNlpRnn(object):
    def __init__(self, config):
        self.config = config

        with tf.name_scope("Embeddings"):
            self.embeddings = tf.compat.v1.get_variable("InputEmbeddings", [self.config.language.input_vocab_size, self.config.language.input_emb_size],
                                              dtype=tf.float32)

        with tf.name_scope("RNN"):
            with tf.variable_scope("rnn"):
                self.sequences_rnn_cell = cell(self.config.encoder.rnn_size, self.config.encoder.cell_type,
                                               proj=self.config.encoder.proj_layers,
                                               dropout=self.config.encoder.keep_prob,
                                               config=config)

    def __call__(self, inputs, initial_state=None):
        self.sequences = inputs
        self.wes = tf.nn.embedding_lookup(self.embeddings,
                                          self.sequences)  # batch_size x sentence_max_len x word_emb_size

        outputs, state = tf.nn.dynamic_rnn(cell=self.sequences_rnn_cell,
                                           inputs=self.wes,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        return state, outputs


class RnnLm(object):
    def __init__(self, config, name="RnnLm", reuse=False):
        self.config = config
        self.name = name
        self.reuse = reuse

        with tf.variable_scope(self.name, reuse) as scope:
            with tf.variable_scope("ConditionalEmbeddings", reuse=reuse):
                self.add_contextual_embeddings(self.config, reuse)

            self.rnn = SimpleNlpRnn(config)  # rnn used in get_logits used to predict the next token
            self.rnn_scope = scope
            self.proj = tf.layers.Dense(self.config.language.output_emb_size, use_bias=True)

    def add_contextual_embeddings(self, config, reuse):
        """
        Defines contextual embeddings specified in
        the config.
        :param config: Config object.
        :param reuse: Bool value, to reuse or not existing variables
        :return: None. It just initialize some variables
        """
        self.a = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        if config.use_author_embs:
            with tf.variable_scope("AuthorEmbeddings", reuse=reuse):
                self.author_embs =  tf.compat.v1.get_variable(
                    "Embeddings",
                    shape=[config.author_vocab_size, config.author_emb_size],
                    dtype=tf.float32
                )

        self.f = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        if config.use_family_embs:
            with tf.variable_scope("FamilyEmbeddings", reuse=reuse):
                self.family_embs =  tf.compat.v1.get_variable(
                    "Embeddings",
                    shape=[config.family_vocab_size, config.family_emb_size],
                    dtype=tf.float32
                )

        self.is_prose = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        if config.use_is_prose_embs:
            with tf.variable_scope("IsProseEmbeddings", reuse=reuse):
                self.is_prose_embs =  tf.compat.v1.get_variable(
                    "Embeddings",
                    shape=[2, config.is_prose_emb_size],
                    dtype=tf.float32
                )

    def get_contextual_embeddings(self, author, family, is_prose, context_state):
        """
        Actually computes the specified embeddings, and concats them
        to the context_state.
        :param context_state: [batch_size x state_size]
        :param inputs: an iterable of placeholders.
        :return: the updated context state (it self in case there are no
        contextual embeddings).
        """
        if self.config.use_author_embs:
            aes = tf.nn.embedding_lookup(self.author_embs, author)
            aes = tf.tile(aes, multiples=[tf.shape(context_state)[0] // tf.shape(aes)[0], 1])
            context_state = tf.concat((context_state, aes), axis=1)

        if self.config.use_family_embs:
            fes = tf.nn.embedding_lookup(self.family_embs, family)
            fes = tf.tile(fes, multiples=[tf.shape(context_state)[0] // tf.shape(fes)[0], 1])
            context_state = tf.concat((context_state, fes), axis=1)

        if self.config.use_is_prose_embs:
            ispres = tf.nn.embedding_lookup(self.is_prose_embs, is_prose)
            ispres = tf.tile(ispres, multiples=[tf.shape(context_state)[0] // tf.shape(ispres)[0], 1])
            context_state = tf.concat((context_state, ispres), axis=1)

        return context_state

    def __call__(self, x):
        with tf.name_scope(self.name):
            _, states = self.rnn(x)  # get a vector for each token in the input
            self.states = tf.reshape(states, [-1, self.config.encoder.rnn_size])  # (batch_size x sentence_max_len) x encoder_rnn_size
            conditional_states = self.get_contextual_embeddings(self.a, self.f, self.is_prose, self.states)
            proj_conditional_states = self.proj(conditional_states, scope=self.rnn_scope)
            self.logits = tf.matmul(proj_conditional_states, tf.transpose(self.rnn.embeddings))  # (batch_size x sentence_max_len) x vocab_size

        return self.logits


class LanguageModel(object):
    def __init__(self, x, config, name="LanguageModel", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.rnn_lm = RnnLm(config, reuse=reuse)

            self.x = x
            self.config = config
            _go = tf.expand_dims(tf.tile([self.config.language._GO], [tf.shape(x)[0]]), axis=1)
            _x, y = tf.concat((_go, self.x[:, :-1]), axis=1), self.x

            logits = self.rnn_lm(_x)
            batch_size = tf.shape(_x)[0]
            self.preds = tf.reshape(tf.argmax(logits, axis=1), [batch_size, -1])

            # prob distribution over words in vocabulary
            self.p_w = tf.nn.softmax(tf.reshape(logits, shape=[-1, tf.shape(_x)[1], self.config.language.input_vocab_size]))

        with tf.variable_scope("Cost", reuse=reuse):
            self._y = tf.reshape(y, [-1])
            self.pad_mask = tf.cast(tf.not_equal(self._y, tf.ones_like(self._y) * self.config.language._PAD), tf.float32)
            self.unk_mask = tf.cast(tf.not_equal(self._y, tf.ones_like(self._y) * 0), tf.float32)
            self.tg_mask = self.pad_mask * self.unk_mask

            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y, logits=logits)
            self.lm_loss = tf.reduce_sum(ce * self.tg_mask) / tf.reduce_sum(self.tg_mask)
            self.ppl = tf.exp(self.lm_loss)

            if not config.is_test:
                with tf.name_scope("AdamOptimization"):
                    optimizer = tf.train.AdamOptimizer(config.learning.lr)
                    self.train_op = optimizer.minimize(self.lm_loss)

    def run_train_op(self, sess, b, train_summary):
        """
        Runs a train step.
        :param sess: tf.Session instance.
        :param b: input batch tuple of sequences. 2-d tensor of size [batch_size, input_max_seq_len].
        :param train_summary: tf.Summary for training.
        :return: Train op output, value for train summary, loss on the batch and perplexity ppl.
        """
        bx, ba, bf, bispr = b
        feed_dict = {self.x: bx}

        if self.config.use_author_embs:
            feed_dict[self.rnn_lm.a] = ba

        if self.config.use_family_embs:
            feed_dict[self.rnn_lm.f] = bf

        if self.config.use_is_prose_embs:
            feed_dict[self.rnn_lm.is_prose] = bispr

        run_output = sess.run((self.train_op, train_summary, self.lm_loss, self.ppl), feed_dict=feed_dict)
        run_output_dict = {
            "summaries": run_output[1],
            "loss": run_output[2],
            "to_optimize_loss": run_output[2],
            "ppl": run_output[3],
        }

        return run_output_dict

    def val_op(self, sess, v, val_summary):
        """
        Evaluates the loss on vx and vy.
        :param sess: tf.Session instance.
        :param v: input batch sequences. 2-d tensor of size [batch_size, input_max_seq_len].
        :param val_summary: tf.Summary for tracing validation performances.
        :return: value for validation summary, predictions, loss on the batch and perplexity ppl.
        """
        vx, va, vf, vispr = v

        feed_dict = {self.x: vx}

        if self.config.use_author_embs:
            feed_dict[self.rnn_lm.a] = va

        if self.config.use_family_embs:
            feed_dict[self.rnn_lm.f] = vf

        if self.config.use_is_prose_embs:
            feed_dict[self.rnn_lm.is_prose] = vispr

        run_output = sess.run((val_summary, self.lm_loss, self.ppl), feed_dict=feed_dict)
        run_output_dict = {
            "summaries": run_output[0],
            "loss": run_output[1],
            "to_optimize_loss": run_output[1],
            "ppl": run_output[2],
        }

        return run_output_dict

    @staticmethod
    def get_params(FLAGS):
        language_cf = BasicLanguageConfig(vocab_size=FLAGS.vocab_size, input_emb_size=FLAGS.emb_size, input_seq_max_len=FLAGS.input_seq_max_len,
                                          output_seq_max_len=FLAGS.output_seq_max_len)

        encoder_cf = EncoderConfig(FLAGS.enc_size, cell_type=FLAGS.rnns_cell_type, keep_prob=FLAGS.enc_keep_prob, is_test=False)

        learning_cf = LearningConfig(optimizer=tf.train.AdamOptimizer, n_epochs=FLAGS.n_epochs, lr=FLAGS.lr, norm_clip=FLAGS.norm_clip,
                                     max_no_improve=FLAGS.max_no_improve, batch_size=FLAGS.batch_size)

        config = LMConfigs(language_cf, encoder_cf, learning_cf,
                           is_generation=False, is_test=False, restore_model=FLAGS.restore_model)

        config.use_author_embs = FLAGS.use_author_embs
        config.author_vocab_size = FLAGS.author_vocab_size
        config.author_emb_size = FLAGS.author_emb_size

        config.use_family_embs = FLAGS.use_family_embs
        config.family_vocab_size = FLAGS.family_vocab_size
        config.family_emb_size = FLAGS.family_emb_size

        config.use_is_prose_embs = FLAGS.use_is_prose_embs
        config.is_prose_emb_size = FLAGS.is_prose_emb_size

        is_test = True
        val_encoder_cf = copy.copy(config.encoder)
        val_encoder_cf.setIsTest(is_test)

        val_config = LMConfigs(language_cf, val_encoder_cf, learning=None, is_generation=False,
                               is_test=is_test, restore_model=FLAGS.restore_model)

        val_config.use_author_embs = FLAGS.use_author_embs
        val_config.author_vocab_size = FLAGS.author_vocab_size
        val_config.author_emb_size = FLAGS.author_emb_size

        val_config.use_family_embs = FLAGS.use_family_embs
        val_config.family_vocab_size = FLAGS.family_vocab_size
        val_config.family_emb_size = FLAGS.family_emb_size

        val_config.use_is_prose_embs = FLAGS.use_is_prose_embs
        val_config.is_prose_emb_size = FLAGS.is_prose_emb_size

        return config, val_config, val_config

