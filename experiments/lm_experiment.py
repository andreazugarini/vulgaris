import os
import tensorflow as tf
import numpy as np
from utils.basic_utils import save_data, print_and_write


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('val_size', 200, 'size of the validation')
tf.app.flags.DEFINE_integer('test_size', 200, 'size of the validation')


class Experiment(object):
    """
    Generic class to handle standard ML experiments.
    """

    def __init__(self, model, model_val, model_gen, dataset, exp_path, config):
        self.model = model
        self.model_val = model_val
        self.model_gen = model_gen
        self.dataset = dataset
        self.exp_path = exp_path
        self.save_path = os.path.join(self.exp_path, "model.ckpt")

        self.summary_writer_path = os.path.join(self.exp_path, 'summary')
        self.summary_writer = tf.summary.FileWriter(self.summary_writer_path)

        self.config = config

    @staticmethod
    def get_config_proto():
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        return config_proto

    @staticmethod
    def save_model(sess, logs, global_step, saver, save_path):
        """
        Model Checkpoint
        :param sess: tf.Session
        :param logs: log file
        :param global_step: integer
        :param saver: tf.Saver() object
        :param save_path: file path where to save the checkpoint
        :return:
        """
        print_and_write(logs, "\nSaving Best Model at step " + str(global_step) + "\n")
        saver.save(sess, save_path)

    @staticmethod
    def restore_model(logs, sess, restore_path, var_list=None):
        # Restore variables from pretrained model (restore path)
        print_and_write(logs, "Restoring Model in " + str(restore_path))
        sess.run(tf.global_variables_initializer())
        var_list = tf.global_variables() if var_list is None else var_list
        restorer = tf.train.Saver(var_list=var_list)
        restorer.restore(sess, restore_path)
        # sess.run(tf.variables_initializer(op))

    @staticmethod
    def early_stop(logs, stop_cond, max_no_improve):
        if stop_cond >= max_no_improve:
            print_and_write(logs, "\nEARLY STOPPING\n")
            return True

        return False

    @staticmethod
    def checkpoint_manager(sess, logs, loss, best_loss, best_step, stop_cond, restore_cond, global_step, saver, save_path, max_restore_cond=10):
        """
        Handles savings of best models and restoring of previous ones
        when performances do not improve anymore
        :param sess: tf.Session() object
        :param logs: an opened file where to save logs
        :param loss: current loss value
        :param best_loss: best loss value
        :param best_step: step of the best_loss
        :param stop_cond: Number of steps, without improvements
        :param restore_cond: Number of steps without improvements since the last restore
        :param global_step: current step
        :param saver: tf.train.Saver() object
        :param save_path: path where to save the model
        :param max_restore_cond: number of steps before restore
        :return:
        """
        # Model Checkpoint, only if improved
        if loss < best_loss:
            Experiment.save_model(sess, logs, global_step, saver, save_path)
            stop_cond, restore_cond = 0, 0
            best_step = global_step
        else:
            stop_cond += 1
            restore_cond += 1
            if restore_cond%max_restore_cond == 0:
                Experiment.restore_model(logs, sess, save_path)

        # update best score ppl
        best_loss = loss if best_loss > loss else best_loss
        return best_loss, stop_cond, restore_cond, best_step

    @staticmethod
    def add_visualization_summaries(name, tensor):
        with tf.name_scope("Summaries"):
            return tf.summary.scalar(name=name, tensor=tensor)

    def get_initial_exp_conds(self):
        return 1e12, 0, 0, self.config.learning.max_no_improve

    def run(self):
        return NotImplementedError

    def eval(self):
        return NotImplementedError

    def _updt_train_logs(self, logs, batch_outputs, step):
        for summary in batch_outputs["summaries"]:
            self.summary_writer.add_summary(summary, step)

    def _updt_val_logs(self, logs, val_outputs, step):
        summaries = [o["ppl"] for o in val_outputs]
        val_ppl = np.mean(np.array(summaries))
        summary = tf.Summary()
        summary.value.add(tag="val ppl", simple_value=val_ppl)

        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        self.summary_writer.add_summary(summary, step)

    def render(self, logs, examples, preds, gen_size):
        return NotImplementedError

    def _get_val_batch(self, val_size):
        return NotImplementedError

    def _generation_op(self, sess, gen_batch):
        return self.model_gen.gen_op(sess, gen_batch)

    def _validation_op(self, sess, logs, val_summaries, step, val_size, best_loss, best_step, stop_cond, restore_cond, saver):
        """
        Runs the model on the validation (or part)
        :param sess: tf.Session() object
        :param logs: an opened file where to save logs
        :param val_summaries: a list of summaries for validation
        :param step: current time step
        :param val_size: number of examples from the validation to assess
        :param best_loss: current best loss value
        :param best_step: current best step
        :param stop_cond: Number of steps, without improvements
        :param restore_cond: Number of steps without improvements since the last restore
        :param saver: tf.train.Saver() object
        :return: Updated values of best_loss, stop_cond, restore_cond, best_step
        """

        print_and_write(logs, "STEP:" + str(step))
        print_and_write(logs, "\n\n VALIDATION EVALUATION \n\n")

        start = 0
        val_outputs = []
        for j in range(len(self.dataset.val_x)//val_size):
            val_batch = self._get_val_batch((start, start+val_size))
            val_outputs.append(self.model_val.val_op(sess, val_batch, val_summaries))
            start += val_size

        self._updt_val_logs(logs, val_outputs, step)

        val_loss = np.mean(np.array([o["to_optimize_loss"] for o in val_outputs]))
        best_loss, stop_cond, restore_cond, best_step = self.checkpoint_manager(sess, logs, val_loss, best_loss,
                                                                                best_step, stop_cond, restore_cond, global_step=step,
                                                                                saver=saver, save_path=self.save_path)

        return best_loss, stop_cond, restore_cond, best_step

    def _run_train_session(self, logs, train_summaries, val_summaries, val_size):
        saver = tf.train.Saver()
        with tf.Session(config=self.get_config_proto()) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            if self.config.restore_model:
                restore_path = os.path.join(self.config.restore_path, "model.ckpt")
                self.restore_model(logs, sess, restore_path)

            tf.summary.FileWriter(self.summary_writer_path, sess.graph)

            self.dataset.initialize(sess)

            step = 0
            best_step = -1
            best_loss, stop_cond, restore_cond, max_no_improve = self.get_initial_exp_conds()  # early stops conditions
            batch_size = self.config.learning.batch_size

            for e in range(self.config.learning.n_epochs):
                print_and_write(logs, "Epoch: " + str(e))
                for batch in self.dataset.get_batches(batch_size):
                    batch_output = self.model.run_train_op(sess, batch, train_summaries)  # train step

                    if step % 100 == 0:
                        self._updt_train_logs(logs, batch_output, step)
                    step += 1

                best_loss, stop_cond, restore_cond, best_step = self._validation_op(sess, logs, val_summaries, step, val_size,
                                                                                    best_loss, best_step, stop_cond, restore_cond,
                                                                                    saver)

                if self.early_stop(logs, stop_cond, max_no_improve):
                    break

            self.callback_hooks(sess)
        return best_loss, best_step

    def callback_hooks(self, sess):
        pass


class LMExperiment(Experiment):
    """
    Language Modeling Extension of Experiment class.
    """
    def __init__(self, model, model_val, model_gen, dataset, exp_path, config):
        super().__init__(model, model_val, model_gen, dataset, exp_path, config)

    @staticmethod
    def get_loss(output):
        """Picks the proper position from an iterable of elements"""
        return output

    @staticmethod
    def get_preds(output):
        """Picks the proper position from an iterable of elements"""
        return output

    def _updt_train_logs(self, logs, batch_outputs, step):
        super()._updt_train_logs(logs, batch_outputs, step)

        print_and_write(logs, "\nLoss: " + str(batch_outputs["loss"]))
        print_and_write(logs, " PPL: " + str(batch_outputs["ppl"]))

    def _updt_val_logs(self, logs, val_outputs, step):
        super()._updt_val_logs(logs, val_outputs, step)

        print_and_write(logs, "\nValidation Loss: " + str(np.mean(np.array([o["loss"] for o in val_outputs]))))
        print_and_write(logs, " Validation PPL: " + str(np.mean(np.array([o["ppl"] for o in val_outputs]))) + "\n")

    def render(self, logs, examples, preds, gen_size):
        pass

    def _get_val_batch(self, val_size):
        start, end = val_size
        return self.dataset.val_x[start:end], self.dataset.val_a[start:end], self.dataset.val_f[start:end], self.dataset.val_ispr[start:end]

    def set_hooks(self):
        pass

    def callback_hooks(self, sess):
        self.store_author_embeddings(sess)
        self.store_family_embeddings(sess)

    def store_author_embeddings(self, sess):
        if self.config.use_author_embs:
            authors = sess.run(self.model.rnn_lm.author_embs)
            save_data(self.dataset.authors_vocabulary, os.path.join(self.exp_path, "authors_vocabulary.pkl"))
            save_data(authors, os.path.join(self.exp_path, "authors_embeddings.pkl"))

    def store_family_embeddings(self, sess):
        if self.config.use_family_embs:
            family = sess.run(self.model.rnn_lm.family_embs)
            save_data(self.dataset.families_vocabulary, os.path.join(self.exp_path, "families_vocabulary.pkl"))
            save_data(family, os.path.join(self.exp_path, "family_embeddings.pkl"))

    def run(self):
        logs = os.path.join(self.exp_path, "train_logs.txt")
        logs = open(logs, "w")

        val_summaries = []
        # Summaries
        train_summary = self.add_visualization_summaries("Train PPL", self.model.ppl)
        val_summary = self.add_visualization_summaries("Valid PPL", self.model_val.ppl)

        val_summaries.append(val_summary)

        best_loss, best_step = self._run_train_session(logs, train_summaries=[train_summary], val_summaries=val_summaries,
                                                       val_size=FLAGS.val_size)

        return best_loss, best_step

    @staticmethod
    def _get_test_batch(dataset, test_size):
        start, end = test_size
        return dataset.val_x[start:end], dataset.val_a[start:end], dataset.val_f[start:end], dataset.val_ispr[start:end]

    def test(self, dataset):
        """
        Evaluate PPL on a given dataset
        :param dataset: An instance of LMDataset
        :return: PPL of the model in the dataset
        """
        logs = os.path.join(self.exp_path, "test_logs.txt")
        logs = open(logs, "w")

        val_summary = self.add_visualization_summaries("Test PPL", self.model_val.ppl)
        val_summaries = [val_summary]

        test_batch_size = FLAGS.test_size
        with tf.Session(config=self.get_config_proto()) as sess:
            restore_path = os.path.join(self.config.restore_path, "model.ckpt")
            self.restore_model(logs, sess, restore_path)
            val_outputs = []
            start = 0
            for j in range(len(dataset.val_x)//test_batch_size):
                val_batch = self._get_test_batch(dataset, (start, start+test_batch_size))
                val_outputs.append(self.model_val.val_op(sess, val_batch, val_summaries))
                start += test_batch_size

        summaries = [o["ppl"] for o in val_outputs]
        val_ppl = np.mean(np.array(summaries))

        return val_ppl
