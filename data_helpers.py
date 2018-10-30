import re

import gc
import numpy as np
import pandas as pd


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def calc_review_score(data):
    try:
        if data['desc_scr'] + data['lgst_scr'] + data['serv_scr'] > 13:
            return [1, 0]
        elif data['desc_scr'] < 3 or data['lgst_scr'] < 3 or data['serv_scr'] < 3:
            return [0, 1]
        else:
            return [1, 0]
    except:
        return [1, 0]


def load_review_data_and_labels(path):
    df = pd.read_csv(path, sep='\t', lineterminator='\n')
    df = df.dropna(axis=0, how='any')
    df['label'] = df.apply(calc_review_score, axis=1)
    x_text, y = list(df['words']), np.array(list(df['label'])).astype(np.float32)
    del df
    gc.collect()
    return x_text, y


def build_word2vec_matrix(vocab_processor, vector_size, word2vec_service):
    matrix = np.zeros((len(vocab_processor.vocabulary_), vector_size))
    for word, word_id in vocab_processor.vocabulary_._mapping.items():
        matrix[word_id] = word2vec_service.get_word_vector(word)
    return matrix.astype(np.float32)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def chinese_tokenizer(docs):
    for doc in docs:
        yield list(doc.split(' '))
