import numpy as np
import glob
import os
from random import randint
from six.moves import cPickle as pickle
from music21 import interval, pitch, instrument, note, stream, chord, converter, analysis, key
from tqdm import tqdm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.optimizers import Adadelta, RMSprop



def load_dataset(dataset_path):
    """Load notes from all midi files in `dataset_path`
    :param dataset_path: path to folder containing the midi dataset
        (eg. datasets/JSB Chorales). NOTE: Don't include `/` at the end of pathname
    :return: list of all music21.stream.Score scores in the dataset
    """

    scores = []
    for midi_file in tqdm(glob.glob(dataset_path+'/*.mid')):
        s = converter.parse(midi_file)
        scores.append(s)
    return scores


def preprocess_score(s, instr=None):
    """Preprocess a score `s` to facilitate training by:
        - Transposing each key to C
        - Change time signature
        - Quantize the score
        - Ignore notes faster than sixteenth notes
        - Using exclusively parts with `instr` as instrument
    :param s: a music21.stream.Score score
    :param instr: music21.instrument.Instrument to predict notes for
                Every instrument in a score has a different playing style
    """

    s = s.flat.notes


    # Ignore notes faster than a sixteenth note
    for n in s.recurse(skipSelf=True):
        if n.duration.quarterLength < 0.25:
            s.remove(n)


    # Transpose to C
    try:
        k = s.analyze('key')
    # Key for score `s` can't be found
    except analysis.discrete.DiscreteAnalysisException as e:
        print(e)
        return

    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    s = s.transpose(i)

    # Snap notes to sixteenth and eighth triplets, whichever is closer
    # (ie make sure each note has perfect timing)
    s = s.quantize()

    return s


def encode_score(s):
    """Encode pitches in a score `s` in order to more easily fed them
    to the model. Map each music21.note.Note note to its pitch string, and
    each music21.note.Chord chord to a string of the composing pitches
    separated by ` `. These are then going to be the target classes
    for the multi-class classification problem
    :param s: a music21.stream.Score score
    """

    encoded_score = [] # List of preprocessed notes from score `s`

    for n in s:
        if n.isChord:
            encoded_score.append(' '.join(str(x) for x in n.normalOrder))
        else:
            encoded_score.append(str(n.pitch))

    return encoded_score


def get_sequences(prep_notes, seq_len, n_classes):
    """Prepare input and output sequences to be fed to the model
    :param prep_notes: list of all preprocessed notes
    :param seq_len: length of the input sequence used to predict a note
    """

    # X, input sequences
    # Y, target classes, one for each sequence in X
    X, Y = [], []

    # Create input sequences with corresponding outputs
    for i in tqdm(range(0, len(prep_notes) - seq_len)):
        X.append(prep_notes[i:i + seq_len])
        Y.append(prep_notes[i + seq_len])

    n_seq = len(X) # number of input sequences

    # Given a sequence of length `seq_len` the model will try to
    # predict the next element of the sequence

    # Reshape the input to match input layer and normalize
    X = np.reshape(X, (n_seq, seq_len, 1))
    X_norm = X / float(n_classes)

    Y = np_utils.to_categorical(Y)

    return X, X_norm, Y


def build_model(X, n_classes, dropout_rate=0.3, weights_filepath=None):
    """Build the RNN using Keras
    :param X: input sequences
    :param n_classes: Number of classes the model has to classify
    :param dropout_rate: the fraction of input nodes dropped by dropout
    :param weights_filepath: path to hdf5 file with previously trained weights.
            allows to continue training from there.
            NOTE: If `None`, no weights are going to be loaded
    """

    # Architecture from here https://arxiv.org/pdf/1604.08723.pdf
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(512))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))


    # Best optimizers for RNNs suggested here
    # http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/

    opt = RMSprop(lr=.002, decay=0.95)
    opt = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    if weights_filepath:
        model.load_weights(weights_filepath)

    return model


def train(model, X, Y, n_epochs, batch_size, checkpoint_path):
    """Train the model
    :param model: the model in Keras
    :param X: input sequences
    :param Y: output sequences
    :param n_epochs: number of epochs to train for
    :param batch_size: minibatch size
    """

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    model.fit(X, Y, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks)



















def generate_notes(model, starting_seq, n_classes, all_pitches, n_notes):
    """Get the model to predict notes, given sequences.
    Given a sequence, the model looks up the probability distribution
     use that to pick the pitch (or class) with the highest probability.
    Starting `starting_seq`, generate a note and append the generated
     note to the current input sequence `curr_seq` and remove the first note
     from `curr_seq`. Then use `curr_seq` as the next input to the model
    :param model: The model built in Keras
    :param starting_seq: The starting sequence
    :param n_classes: The number of distinct pitches (or classes) in the dataset
    :param all_pitches: All distinct pitches in the datset
    :param n_notes: Number of notes to be predicted
    :return: List of notes predicted by the model
    """

    decode_note = dict((code, note_) for code, note_ in enumerate(all_pitches))

    curr_seq = starting_seq[:]
    out = []

    for i in range(n_notes):
        seq_in = np.reshape(curr_seq, (1, len(curr_seq), 1))
        seq_in = seq_in / float(n_classes)

        # Predict the next note
        pred = model.predict(seq_in, verbose=0)

        # Decode note at index with the highest probability
        note_idx = np.argmax(pred)
        note_ = decode_note[note_idx]
        out.append(note_)

        # Append predicted note to the `curr_seq`, and remove
        # first note in `curr_seq`
        curr_seq = np.vstack((curr_seq[1:], [note_idx]))

    return out


def decode_output(s):
    """Doing the opposite of the method `encode_score()`. Decodes `s`
    and builds a music21.stream.Stream with the model's predictions.
    The stream will have notes of fixed duration
    :param s: a list of classes (or pitches) that the model has predicted
    :return: a music21.stream.Stream stream of music21.note.Note notes
    """

    pos = 0 # position of the current note
    duration = 0.5 # duration of each note
    out = []

    # Decode notes to music21.note.Note object and give each 0.5 duration
    for x in s:
        if (' ' in x) or x.isdigit(): # x is a chord
            chord_notes = []
            for pitch_ in x.split(' '):
                note_ = note.Note(int(pitch_))
                note_.storedInstrument = instrument.Piano()
                chord_notes.append(note_)
            chord_ = chord.Chord(chord_notes)
            chord_.offset = pos
            out.append(chord_)
        else: # x is a note
            note_ = note.Note(x)
            note_.offset = pos
            note_.storedInstrument = instrument.Piano()
            out.append(note_)
        pos += duration

    return stream.Stream(out)


def add_variability(str):
    """Preprocessing performed by `preprocess_score()` made all scores pretty much
    similar to each other. This method adds some variability to the model's prediction
    :param str: music21.stream.Stream stream of predicted notes
    """

    try:
        k = str.analyze('key')
    # Key for score `str` can't be found
    except analysis.discrete.DiscreteAnalysisException as e:
        print(e)
        return

    most_common_keys = list(map(lambda x: key.Key(x), ['G', 'C', 'D', 'A']))
    i = interval.Interval(k.tonic, most_common_keys[randint(0, len(most_common_keys)-1)].tonic)
    return str.transpose(i)


def write_midi(out_stream, filename):
    """Create a midi file called `filename` from a music21.stream.Stream `out_stream`
    :param out_stream: a music21.stream.Stream stream
    :param filename: the name of the midi file to be created
    """
    out_stream.write('midi', fp=filename)




# http://www-etud.iro.umontreal.ca/~boulanni/icml2012

def run(dataset_path, weights_filepath=None):
    folder_name = dataset_path.split('/')[-2]
    pickle_path = 'pickles/'+folder_name+'.pickle'


    # Refer to this to pickle/unpickle scores
    # http://web.mit.edu/music21/doc/moduleReference/moduleFreezeThaw.html

    if os.path.isfile(pickle_path):
        with open(pickle_path, 'rb') as f:
            notes = pickle.load(f)
    else:
        scores = load_dataset(dataset_path)

        # Preprocess and encode all scores in the dataset
        scores = list(map(lambda s: encode_score(preprocess_score(s)), scores))

        # Concatenate all notes in `scores` into a single list
        flatten = lambda l: [item for sublist in l for item in sublist]
        notes = flatten(scores)

        with open(pickle_path, 'wb') as f:
            pickle.dump(notes, f)


    # All pitches occurring in the dataset
    all_pitches = sorted(set(notes))

    # Encode all pitches in the dataset
    # NOTE: The name `note_` is often used throughout the code to
    # avoid shadowing the `note` namespace imported from music21
    encode_pitch = dict((note_, code) for code, note_ in enumerate(all_pitches))
    encoded_pitches = list(map(lambda x: encode_pitch[x], notes))


    # Number of different classes that the model is classifying
    n_classes = len(all_pitches)

    X, X_norm, Y = get_sequences(encoded_pitches, seq_len=16, n_classes=n_classes)
    model = build_model(X_norm, n_classes, 0.3, weights_filepath)

    # Batch size of 64 suggested here
    # https://arxiv.org/pdf/1604.08723.pdf
    if not weights_filepath: # If weights are not given, train
        train(model, X_norm, Y, 10, 64, folder_name+"-softmax-{epoch:02d}-{loss:.4f}.hdf5")

    starting_seq = X[randint(0, len(X)-1)]
    out = generate_notes(model, starting_seq, n_classes, all_pitches, 500)
    out_stream = decode_output(out)
    #out_stream = add_variability(out_stream)
    write_midi(out_stream, folder_name+'-out.mid')



run('datasets/ff/train')

