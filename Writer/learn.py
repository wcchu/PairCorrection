import tensorflow as tf
import logging
# import numpy as np
import collections
import os
import pickle

# model
CHECKPOINT_DIR = 'checkpoints'
EMBEDDING_SIZE = 128
RNN_UNITS = 512

# training
DATA_DIR = "bible_1000.txt"  # TODO: change back to full data after development
EPOCHS = 1  # TODO: change back to 3 after development
TIME_STEPS = 200
BATCH_SIZE = 64
BUFFER_SIZE = 10000


def get_data(data_path=None):
    """Load raw data from data directory "data_path".
    Reads text file, converts strings to integer ids
    Args:
    data_path: string path to the directory
    Returns:
    tuple (raw_data, vocabulary)
    """

    data = list(open(data_path, "r").read())
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    chars, _ = list(zip(*count_pairs))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}

    data_in_ids = [char_to_id[char] for char in data]
    return data, data_in_ids, char_to_id, id_to_char


def input_and_response(example):
    '''Define input and response parts of each example'''
    example_input = example[:-1]
    example_response = example[1:]
    return example_input, example_response


def preprocess(d):
    '''Preprocess data'''
    # tf.Dataset
    ds = tf.data.Dataset.from_tensor_slices(d)

    # create examples
    # NOTE: the "batch" defined here is one example (batch of characters)
    # instead of batch of examples
    examples = ds.batch(TIME_STEPS + 1, drop_remainder=True)

    # prepare training dataset
    dataset = examples.map(input_and_response).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)
    return dataset


def build_model(n_chars, emb_size, rnn_units, batch_size):
    '''Define keras rnn model'''
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(n_chars,
                                  emb_size,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n_chars)
    ])


def loss(responses, logits):
    '''Loss function'''
    return tf.keras.losses.sparse_categorical_crossentropy(responses,
                                                           logits,
                                                           from_logits=True)


def train_model(data, nc):
    '''Train model with given training data
    data: training data (tf.data.Dataset object)
    nc: number of distinct characters

    The result of training is saved in checkpoint dir as weights.
    The function doesn't return any object.
    '''
    # create model
    model = build_model(n_chars=nc,
                        emb_size=EMBEDDING_SIZE,
                        rnn_units=RNN_UNITS,
                        batch_size=BATCH_SIZE)

    # compile model with optimizer and loss function
    model.compile(optimizer='adam', loss=loss)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)

    # Execute training
    model.fit(data, epochs=EPOCHS, callbacks=[checkpoint_callback])
    model.summary()


def run():

    # import training data
    raw_data_chars, raw_data_ids, char_to_id, id_to_char = get_data(DATA_DIR)
    n_chars = len(char_to_id)
    print("number of distinct characters = {}".format(n_chars))

    # preprocess data
    ds_train = preprocess(raw_data_ids)

    # build and train model
    train_model(ds_train, n_chars)

    # pickle
    pickle.dump(n_chars, open('n_chars.pk', 'wb'))
    pickle.dump(char_to_id, open('char_to_id.pk', 'wb'))
    pickle.dump(id_to_char, open('id_to_char.pk', 'wb'))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)
    run()
