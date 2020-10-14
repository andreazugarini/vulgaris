import collections
import numpy as np


class Vocabulary(object):
    def __init__(self, vocab_size=None):
        self.dictionary = dict()
        self.rev_dictionary = dict()
        self.count = []
        self.special_tokens = []
        self.vocab_size = vocab_size

    def build_vocabulary_from_counts(self, count, special_tokens=[]):
        """
        Sets all the attributes of the Vocabulary object.
        :param count: a list of lists as follows: [['token', number_of_occurrences],...]
        :param special_tokens: a list of strings. E.g. ['<EOS>', '<PAD>',...]
        :return: None
        """

        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)

        # adding eventual special tokens to the dictionary (e.g. <EOS>,<PAD> etc..)
        d = len(dictionary)
        for i, token in enumerate(special_tokens):
            dictionary[token] = d + i

        self.count = count
        self.dictionary = dictionary
        self.rev_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.special_tokens = special_tokens
        self.vocab_size = len(dictionary)

    def build_vocabulary_from_tokens(self, tokens, vocabulary_size=None, special_tokens=[], cutoff_freq=-1):
        """
        Given a list of tokens, it sets the Vocabulary object attributes by constructing
        a dictionary mapping each token to a unique id.
        :param tokens: a list of strings.
         E.g. ["the", "cat", "is", ... ".", "the", "house" ,"is" ...].
         NB: Here you should put all your token instances of the corpus.
        :param vocabulary_size: The number of elements of your vocabulary. If there are more
        than 'vocabulary_size' elements on tokens, it considers only the 'vocabulary_size'
        most frequent ones.
        :param special_tokens: Optional. A list of strings. Useful to add special tokens in vocabulary.
        If you don't have any, keep it empty.
        :return: None
        """

        vocabulary_size = vocabulary_size if vocabulary_size is not None else self.vocab_size
        vocabulary_size = vocabulary_size - (len(special_tokens) + 1) if vocabulary_size else None
        # counts occurrences of each token
        count = [['<UNK>', -1]]
        count.extend(collections.Counter(tokens).most_common(vocabulary_size))  # takes only the most frequent ones, if size is None takes them all
        count = self.cutoff(counts=count, cutoff_freq=cutoff_freq)
        self.build_vocabulary_from_counts(count, special_tokens)  # actually build the vocabulary
        # self._set_unk_count(tokens)  # set the number of OOV instances

    @staticmethod
    def merge_vocabulary(vocab0, vocab1, vocabulary_size=-1):
        """
        Merge two Vocabulary objects into a new one.
        :param vocab0: first Vocabulary object
        :param vocab1: second Vocabulary object
        :param vocabulary_size: parameter to decide the merged vocabulary size.
        With default value -1, all the words of both vocabularies are preserved.
        When set to 0, the size of the vocabulary is set to the size of vocab0,
        when set to 1 it is kept the size of vocab1.
        :return: a new vocabulary
        """
        # get size of the new vocabulary
        vocab_size = vocab0.vocab_size + vocab1.vocab_size if vocabulary_size == -1 else vocabulary_size
        merged_special_tokens = list(set(vocab0.special_tokens) | set(vocab1.special_tokens))

        # merge the counts from the two vocabularies and then selects the most_common tokens
        merged_counts = collections.Counter(dict(vocab0.count)) + collections.Counter(dict(vocab1.count))
        merged_counts = merged_counts.most_common(vocab_size)
        count = [['<UNK>', -1]]
        count.extend(merged_counts)

        # create the new vocabulary
        merged_vocab = Vocabulary(vocab_size)
        merged_vocab.build_vocabulary_from_counts(count, merged_special_tokens)
        return merged_vocab

    def cutoff(self, counts, cutoff_freq=-1):
        cutoff_counts = []
        if cutoff_freq >= 1:
            print(f"============= Cutting off {cutoff_freq}  ===========")
            cutoff_counts.append(['<UNK>', -1])
            for token_count in counts:
                if token_count[1] > cutoff_freq:
                    cutoff_counts.append(token_count)
        else:
            cutoff_counts = counts

        return cutoff_counts

    def get_total_counts(self):
        return sum([c[1] for c in self.count])

    def get_token_counts(self, token):
        return self.count[self.dictionary[token]][1] if token in self.dictionary else 0

    @staticmethod
    def merge_vocabularies(vocab_list, vocab_size=None):
        """
        Join a list of vocabularies into a new one.
        :param vocab_list: a list of Vocabulary objects
        :param vocab_size: the maximum size of the merged vocabulary.
        :return: a vocabulary merging them all.
        """
        vocab_size = vocab_size if vocab_size else sum([v.vocab_size for v in vocab_list])
        merged_vocab = Vocabulary(vocab_size)
        for voc in vocab_list:
            merged_vocab = Vocabulary.merge_vocabulary(merged_vocab, voc, vocab_size)
        return merged_vocab

    def string2id(self, dataset):
        """
        Converts a dataset of strings into a dataset of ids according to the object dictionary.
        :param dataset: any string-based dataset with any nested lists.
        :return: a new dataset, with the same shape of dataset, where each string is mapped into its
        corresponding id associated in the dictionary (0 for unknown tokens).
        """

        def _recursive_call(items):
            new_items = []
            for item in items:
                if isinstance(item, str) or isinstance(item, int) or isinstance(item, float):
                    new_items.append(self.word2id(item))
                else:
                    new_items.append(_recursive_call(item))
            return new_items

        return _recursive_call(dataset)

    def id2string(self, dataset):
        """
        Converts a dataset of integer ids into a dataset of string according to the reverse dictionary.
        :param dataset: any int-based dataset with any nested lists. Allowed types are int, np.int32, np.int64.
        :return: a new dataset, with the same shape of dataset, where each token is mapped into its
        corresponding string associated in the reverse dictionary.
        """
        def _recursive_call(items):
            new_items = []
            for item in items:
                if isinstance(item, int) or isinstance(item, np.int) or isinstance(item, np.int32) or isinstance(item, np.int64):
                    new_items.append(self.id2word(item))
                else:
                    new_items.append(_recursive_call(item))
            return new_items

        return _recursive_call(dataset)

    def word2id(self, item):
        """
        Maps a string token to its corresponding id.
        :param item: a string.
        :return: If the token belongs to the vocabulary, it returns an integer id > 0, otherwise
        it returns the value associated to the unknown symbol, that is typically 0.
        """
        return self.dictionary[item] if item in self.dictionary else self.dictionary['<UNK>']

    def id2word(self, token_id):
        """
        Maps an integer token to its corresponding string.
        :param token_id: an integer.
        :return: If the id belongs to the vocabulary, it returns the string
        associated to it, otherwise it returns the string associated
        to the unknown symbol, that is '<UNK>'.
        """

        return self.rev_dictionary[token_id] if token_id in self.rev_dictionary else self.rev_dictionary[self.dictionary['<UNK>']]

    def get_unk_count(self):
        return self.count[0][1]

    def _set_unk_count(self, tokens):
        """
        Sets the number of OOV instances in the tokens provided
        :param tokens: a list of tokens
        :return: None
        """
        data = list()
        unk_count = 0
        for word in tokens:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary['<UNK>']
                unk_count += 1
            data.append(index)
        self.count[0][1] = unk_count

    def add_element(self, name, is_special_token=False):
        if name not in self.dictionary:
            self.vocab_size += 1
            self.dictionary[name] = self.vocab_size
            self.rev_dictionary[self.vocab_size] = name

            if is_special_token:
                self.special_tokens = list(self.special_tokens)
                self.special_tokens.append(name)

            self.count.append([name, 1])

    def set_vocabulary(self, dictionary, rev_dictionary, special_tokens, vocab_size):
        self.dictionary = dictionary,
        self.rev_dictionary = rev_dictionary
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size