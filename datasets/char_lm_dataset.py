import random

from datasets.lm_dataset import LMDataset
from utils.vocabulary import Vocabulary

random.seed(a=123)
SPECIAL_TOKENS = ["<EOS>", "<EOL>", "<s>"]


class CharLMDataset(LMDataset):
    """
    Char-based extension of LMDataset.
    """
    def __init__(self, config, vocabulary=None, authors_vocabulary=None, families_vocabulary=None, gen_len=1, name="CharLMDataset"):
        super().__init__(config, vocabulary,
                         authors_vocabulary=authors_vocabulary,
                         families_vocabulary=families_vocabulary,
                         gen_len=gen_len,
                         name=name)

    @staticmethod
    def to_chars(word):
        word = [c for c in word] if word not in SPECIAL_TOKENS else [word]
        return word + ["<s>"]

    @staticmethod
    def tokenize(raw_data):
        """
        :param raw_data: a list of strings
        :return: a list of lists of tokens (chars)
        """
        sentences = [s.split() for s in raw_data]

        tokenized_sentences = []
        for s in sentences:
            tokenized_sentence = []
            for w in s:
                tokenized_sentence.extend(CharLMDataset.to_chars(w))
            tokenized_sentences.append(tokenized_sentence[:-1])  # -1 removes extra final separator

        return tokenized_sentences

    def build(self, filename, split_size=0.8, authors=None, collections=None, families=None):
        print("\nBuilding the dataset....")
        examples = self.load(filename, authors, collections, families)
        random.shuffle(examples)
        raw_x, raw_a, raw_f, raw_ispr = zip(*examples)

        # cleanup & tokenize data
        raw_x = self.tokenize([LMDataset.preprocess(ex) for ex in raw_x])

        # dataset split
        self.raw_train_x, self.raw_val_x = LMDataset.split(raw_x, train_size=split_size)
        self.raw_train_a, self.raw_val_a = LMDataset.split(raw_a, train_size=split_size)
        self.raw_train_f, self.raw_val_f = LMDataset.split(raw_f, train_size=split_size)
        self.raw_train_ispr, self.raw_val_ispr = LMDataset.split(raw_ispr, train_size=split_size)

        if self.vocabulary is None:
            # creates vocabulary
            x_tokens = [item for sublist in self.raw_train_x for item in sublist]  # get tokens
            self._create_vocab(x_tokens)
            print(f"Char Vocabulary size: {len(self.vocabulary.rev_dictionary)}")

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

        print("Dataset Built\n")
