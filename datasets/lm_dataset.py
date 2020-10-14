import pandas as pd
import random
import re
from math import floor

from utils.vocabulary import Vocabulary
from utils.basic_utils import pad_list

seed = 123
random.seed(a=seed)


class LMDataset:
    def __init__(self, config, vocabulary=None, authors_vocabulary=None, families_vocabulary=None, gen_len=4, name="LMDataset"):
        self.config = config
        self.name = name
        self.gen_len = gen_len
        self.raw_train_a, self.raw_val_a = None, None
        self.raw_train_f, self.raw_val_f = None, None

        self.vocabulary, self.authors_vocabulary, self.families_vocabulary = vocabulary, authors_vocabulary, families_vocabulary
        self.raw_train_ispr, self.raw_val_ispr = None, None
        self.train_ispr, self.val_ispr = None, None

    def initialize(self, sess):
        pass

    def load(self, filename, authors=None, collections=None, families=None):
        """
        Load raw dataset from a csv file.
        A portion of the data can be sub-selected.
        :param filename: a csv file
        :param authors: a list of strings.
        If provided, only authors in the list are retrieved.
        :param collections: a list of strings of collections.
        If provided, only collections in the list are retrieved.
        :param families: a list of string of families.
        If provided, only families in the list are retrieved.

        :return: a dataset in form of a list of texts
        """
        df = pd.read_csv(filename, encoding="utf-8", sep=";")
        df = df.sample(frac=1, random_state=seed, axis=0).reset_index(drop=True)
        print(df)

        txts, authors, families, text_type = LMDataset.get_txt_from_df(df, authors, collections, families)

        texts = list(txts)
        return LMDataset.create_examples(texts,
                                         authors, families, text_type,
                                         gen_len=self.gen_len,
                                         max_seq_len=self.config.language.output_seq_max_len
                                         )

    @staticmethod
    def create_examples(texts, authors, families, text_type, gen_len=4, max_seq_len=50):
        """
        :param texts: a list of strings with a sequence of verses/sentences.
        :param authors:
        :param families:
        :return: a list of tuples
        """
        eol = " <EOL> "
        eos = " <EOS> "
        t_stanzas = []
        for s_id, text in enumerate(texts):
            author = authors[s_id]
            family = families[s_id]
            is_prose = 1 if text_type[s_id] == "prose" else 0
            if not is_prose:
                lines = text.split(eol.strip())  # list of strings
                for i in range(0, len(lines), gen_len):
                    t_stanzas.append((eol.join(lines[i:i+gen_len]).strip(), author, family, is_prose))
            else:
                lines = text.split(eos.strip())  # list of strings
                for line in lines:
                    words = line.split()
                    for w in range(0, len(words), max_seq_len-1):
                        t_stanzas.append((' '.join(words[w:w+max_seq_len-1] + [eos.strip()]).strip(), author, family, is_prose))

        return t_stanzas

    @staticmethod
    def preprocess(str_stanza):
        str_stanza = str_stanza.lower()
        str_stanza = str.replace(str_stanza, "<eos>", " <EOS> ")
        str_stanza = str.replace(str_stanza, "<eol>", " <EOL> ")
        str_stanza = str.replace(str_stanza, "p.", '')
        str_stanza = str.replace(str_stanza, "Vol.", '')
        str_stanza = str.replace(str_stanza, "vol.", '')
        str_stanza = str.replace(str_stanza, "'", " ' ")
        str_stanza = str.replace(str_stanza, ";", " ; ")
        str_stanza = str.replace(str_stanza, ".", " . ")
        str_stanza = str.replace(str_stanza, ",", " , ")
        str_stanza = str.replace(str_stanza, "!", " ! ")
        str_stanza = str.replace(str_stanza, "?", " ? ")
        str_stanza = str.replace(str_stanza, ":", " : ")
        str_stanza = str.replace(str_stanza, "=", "")
        str_stanza = str.replace(str_stanza, "%", "")
        str_stanza = str.replace(str_stanza, "*", "")
        str_stanza = str.replace(str_stanza, "”", ' \" ')
        str_stanza = str.replace(str_stanza, "“", ' \" ')

        str_stanza = re.sub("\[.*\]", "", str_stanza)
        str_stanza = re.sub("\d+", "", str_stanza)

        return str_stanza

    @staticmethod
    def get_txt_from_df(df, authors=None, collections=None, families=None):
        """
        Filter df rows accordingly to author(s), collections(s) and
        family(ies), then return the list their texts.
        :param df: a pandas Dataframe. We assume it has 'author', 'collection',
        'family' among its columns.
        :param authors: a list
        :param collections: a list
        :param families: a list
        :return: a tuple of texts, authors, families and composition types
        """
        tmp_df = df
        if authors:
            tmp_df = tmp_df[df.author.isin(authors)]

        if collections:
            tmp_df = tmp_df[df.collection.isin(collections)]

        if families:
            tmp_df = tmp_df[df.family.isin(families)]

        return tmp_df["text"].values, tmp_df["author"].values, tmp_df["family"].values, tmp_df["type"].values

    @staticmethod
    def split(raw_data, train_size=0.8):
        if train_size < 0.0:
            return [], raw_data
        elif train_size >= 1.0:
            return raw_data, []
        else:
            size = floor(len(raw_data)*train_size)
        return raw_data[:size], raw_data[size:]


    def build(self, filename, split_size=0.8, authors=None, collections=None, families=None):
        examples = self.load(filename, authors, collections, families)
        random.shuffle(examples)
        raw_x, raw_a, raw_f, raw_ispr = zip(*examples)

        print("================")
        print("Authors\n")
        print(set(raw_a))

        print("================")
        print("Families\n")
        print(set(raw_f))

        print("================")
        print("Text Type\n")
        print(set(raw_ispr))

        # cleanup & tokenize data
        raw_x = self.tokenize([self.preprocess(ex) for ex in raw_x])

        # dataset split
        self.raw_train_x, self.raw_val_x = LMDataset.split(raw_x, train_size=split_size)
        self.raw_train_a, self.raw_val_a = LMDataset.split(raw_a, train_size=split_size)
        self.raw_train_f, self.raw_val_f = LMDataset.split(raw_f, train_size=split_size)
        self.raw_train_ispr, self.raw_val_ispr = LMDataset.split(raw_ispr, train_size=split_size)

        if self.vocabulary is None:
            # creates vocabulary
            x_tokens = [item for sublist in self.raw_train_x for item in sublist]  # get tokens
            self._create_vocab(x_tokens)
            print(f"Vocabulary size: {len(self.vocabulary.rev_dictionary)}")


        if self.authors_vocabulary is None:
            # creates vocabulary
            a_tokens = [item for item in self.raw_train_a]  # get authors vocab
            vocab = Vocabulary(vocab_size=self.config.author_vocab_size)
            vocab.build_vocabulary_from_tokens(a_tokens)
            self.authors_vocabulary = vocab
            print(f"Authors Vocabulary size: {len(self.authors_vocabulary.rev_dictionary)}")

        self.train_a = self.authors_vocabulary.string2id(self.raw_train_a)
        self.val_a = self.authors_vocabulary.string2id(self.raw_val_a)

        if self.families_vocabulary is None:
            # creates vocabulary
            f_tokens = [item for item in self.raw_train_f]  # get family vocab
            vocab = Vocabulary(vocab_size=self.config.family_vocab_size)
            vocab.build_vocabulary_from_tokens(f_tokens)
            self.families_vocabulary = vocab
            print(f"Families Vocabulary size: {len(self.families_vocabulary.rev_dictionary)}")

        self.train_f = self.families_vocabulary.string2id(self.raw_train_f)
        self.val_f = self.families_vocabulary.string2id(self.raw_val_f)

        self.train_ispr = self.raw_train_ispr
        self.val_ispr = self.raw_val_ispr

        # creates x for train
        self.train_x = self._build_dataset(self.raw_train_x, insert_go=False, max_len=self.config.language.seq_max_len, shuffle=False)

        # creates x,for validation
        self.val_x = self._build_dataset(self.raw_val_x, insert_go=False, max_len=self.config.language.seq_max_len, shuffle=False)

        print("TRAINING SET LENGTH: %d\n" % len(self.train_x))
        print("VALIDATION SET LENGTH: %d\n" % len(self.val_x))

    @staticmethod
    def tokenize(raw_data):
        return [x.split() for x in raw_data]

    def _create_vocab(self, tokens, special_tokens=["<PAD>", "<GO>", "<EOS>"]):
        vocab = Vocabulary(vocab_size=self.config.language.input_vocab_size)
        vocab.build_vocabulary_from_tokens(tokens, special_tokens=special_tokens)
        self.vocabulary = vocab

    def _build_dataset(self, raw_data, max_len=100, insert_go=True, keep_lasts=False, pad_right=True, shuffle=True):
        """
        Converts all the tokens in e1_raw_data by mapping each token with its corresponding
        value in the dictionary. In case of token not in the dictionary, they are assigned to
        a specific id. Each sequence is padded up to the seq_max_len setup in the config.

        :param raw_data: list of sequences, each sequence is a list of tokens (strings).
        :param max_len: max length of a sequence, crop longer and pad smaller ones.
        :param insert_go: True to insert <GO>, False otherwise.
        :param keep_lasts: True to truncate initial elements of a sequence.
        :param pad_right: pad to the right (default value True), otherwise pads to left.
        :param shuffle: Optional. If True data are shuffled.
        :return: A list of sequences where each token in each sequence is an int id.
        """
        dataset = []
        for sentence in raw_data:
            sentence_ids = [self.vocabulary.word2id("<GO>")] if insert_go else []
            sentence_ids.extend([self.vocabulary.word2id(w) for w in sentence])
            sentence_ids.append(self.vocabulary.word2id("<EOS>"))
            sentence_ids = pad_list(sentence_ids, self.vocabulary.word2id("<PAD>"), max_len, keep_lasts=keep_lasts, pad_right=pad_right)

            dataset.append(sentence_ids)

        if shuffle:
            return random.sample(dataset, len(dataset))
        else:
            return dataset

    def get_batches(self, batch_size=32):
        """
        Iterator to return random mini-batches during training of a LM
        :param batch_size: int, number of example per mini-batch
        :return: a tuple of examples batch_x, batch_a, batch_f, batch_ispr with:
        the input texts, the authors, the families and the composition kinds, respectively.
        """
        x, a, f, ispr = self.train_x, self.train_a, self.train_f, self.train_ispr

        # Shuffle sentences
        sentences_ids = random.sample(range(len(x)), len(x))

        # Generator for batch
        batch_x, batch_a, batch_f, batch_ispr = [], [], [], []
        if batch_size is None:
            batch_size = len(x)
        for sid in sentences_ids:
            batch_x.append(x[sid])
            batch_a.append(a[sid])
            batch_f.append(f[sid])
            batch_ispr.append(ispr[sid])

            if len(batch_x) % batch_size == 0:
                yield batch_x, batch_a, batch_f, batch_ispr
                batch_x, batch_a, batch_f, batch_ispr = [], [], [], []
