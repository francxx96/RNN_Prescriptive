"""
This script takes as input the workflow, timestamps and an event attribute "resource"
It makes predictions on the workflow & timestamps and the event attribute "resource"

this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite
computationally intensive.

Author: Niek Tax
"""

from __future__ import print_function, division

import copy
import csv
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.optimizers import Nadam, Adam

import shared_variables
from shared_variables import get_unicode_from_int, epochs, folds, validation_split
from training.train_common import create_checkpoints_path, plot_loss


class TrainCFR:
    def __init__(self):
        pass

    @staticmethod
    def _build_model(max_len, num_features, target_chars, target_chars_group, use_old_model):
        print('Build model...')

        main_input = Input(shape=(max_len, num_features), name='main_input')
        processed = main_input

        if use_old_model:
            processed = LSTM(50, return_sequences=True, dropout=0.2)(processed)
            processed = BatchNormalization()(processed)

            activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            activity_output = BatchNormalization()(activity_output)

            group_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            group_output = BatchNormalization()(group_output)

            activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(group_output)
            outcome_output = Dense(1, name='outcome_output')(processed)

            opt = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        else:
            processed = Dense(32//2)(processed)
            processed = BatchNormalization()(processed)
            processed = LeakyReLU()(processed)
            processed = Dropout(0.5)(processed)

            processed = LSTM(64//2, return_sequences=False, recurrent_dropout=0.5)(processed)

            processed = Dense(32//2)(processed)
            processed = BatchNormalization()(processed)
            processed = LeakyReLU()(processed)
            processed = Dropout(0.5)(processed)

            activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(processed)
            outcome_output = Dense(1, name='outcome_output')(processed)
            opt = Adam()

        model = Model(main_input, [activity_output, group_output, outcome_output])
        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy',
                            'outcome_output': 'binary_crossentropy'}, optimizer=opt)
        return model

    @staticmethod
    def _train_model(model, checkpoint_name, X, y_a, y_o, y_g):
        model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        early_stopping = EarlyStopping(monitor='val_loss', patience=7)

        history = model.fit(X, {'act_output': y_a, 'outcome_output': y_o, 'group_output': y_g},
                            validation_split=validation_split, verbose=2, batch_size=32,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=epochs)
        plot_loss(history, os.path.dirname(checkpoint_name))

    @staticmethod
    def train(log_name, models_folder, use_old_model):
        lines = []          # list of all the activity sequences
        lines_group = []    # list of all the resource sequences
        outcomes = []       # outcome of each activity sequence (i.e. each case)

        lastcase = ''
        line = ''
        line_group = ''
        outcome = ''
        first_line = True
        numlines = 0

        path = shared_variables.data_folder / (log_name + '.csv')
        print(path)
        csvfile = open(str(path), 'r')
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)  # skip the headers

        for row in spamreader:  # the rows are "CaseID,ActivityID,CompleteTimestamp,Resource,Label"
            if row[0] != lastcase:  # 'lastcase' is to save the last executed case for the loop
                lastcase = row[0]
                if not first_line:  # here we actually add the sequences to the lists
                    lines.append(line)
                    lines_group.append(line_group)
                    outcomes.append(outcome)
                line = ''
                line_group = ''
                numlines += 1
            line += get_unicode_from_int(row[1])
            line_group += get_unicode_from_int(row[3])
            outcome = row[4]
            first_line = False

        # add last case
        lines.append(line)
        lines_group.append(line_group)
        outcomes.append(outcome)
        numlines += 1

        # separate training data into 2(out of 3) parts
        elements_per_fold = int(round(numlines / 3))

        print("average length of the trace: ", sum([len(x) for x in lines]) / len(lines))
        print("number of traces: ", len(lines))

        fold1 = lines[:elements_per_fold]
        fold1_group = lines_group[:elements_per_fold]
        fold2 = lines[elements_per_fold:2 * elements_per_fold]
        fold2_group = lines_group[elements_per_fold:2 * elements_per_fold]

        lines = fold1 + fold2
        lines_group = fold1_group + fold2_group

        #lines = map(lambda x: x + '!', lines)
        #maxlen = max(map(lambda x: len(x), lines))
        lines = [x + '!' for x in lines]
        maxlen = max([len(x) for x in lines])

        # next lines here to get all possible characters for events and annotate them with numbers
        chars = list(map(lambda x: set(x), lines))  # remove duplicate activities from each separate case
        chars = list(set().union(*chars))   # creates a list of all the unique activities in the data set
        chars.sort()    # sorts the chars in alphabetical order
        target_chars = copy.copy(chars)
        # if '!' in chars:
        chars.remove('!')
        print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        target_char_indices = dict((c, i) for i, c in enumerate(target_chars))

        chars_group = list(map(lambda x: set(x), lines_group))
        chars_group = list(set().union(*chars_group))
        chars_group.sort()
        target_chars_group = copy.copy(chars_group)
        # chars_group.remove('!')
        print('total groups: {}, target groups: {}'.format(len(chars_group), len(target_chars_group)))
        char_indices_group = dict((c, i) for i, c in enumerate(chars_group))
        target_char_indices_group = dict((c, i) for i, c in enumerate(target_chars_group))

        csvfile = open(str(path), 'r')
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader, None)  # skip the headers
        lastcase = ''
        line = ''
        line_group = ''
        outcome = ''
        first_line = True
        lines = []
        lines_group = []
        outcomes = []
        numlines = 0
        for row in spamreader:  # the rows are "CaseID,ActivityID,CompleteTimestamp,Resource,Label"
            # new case starts
            if row[0] != lastcase:
                lastcase = row[0]
                if not first_line:
                    lines.append(line)
                    lines_group.append(line_group)
                    outcomes.append(outcome)
                line = ''
                line_group = ''
                numlines += 1
            line += get_unicode_from_int(row[1])
            line_group += get_unicode_from_int(row[3])
            outcome = row[4]
            first_line = False

        # add last case
        lines.append(line)
        lines_group.append(line_group)
        outcomes.append(outcome)
        numlines += 1

        elements_per_fold = int(round(numlines / 3))

        lines = lines[:-elements_per_fold]
        lines_group = lines_group[:-elements_per_fold]
        lines_o = outcomes[:-elements_per_fold]

        step = 1
        sentences = []
        sentences_group = []
        softness = 0
        next_chars = []
        next_chars_group = []
        #lines = map(lambda x: x + '!', lines)
        #lines_group = map(lambda x: x + '!', lines_group)
        lines = [x + '!' for x in lines]
        lines_group = [x + '!' for x in lines_group]

        sentences_o = []
        for line, line_group, line_o in zip(lines, lines_group, lines_o):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # we add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_group.append(line_group[0:i])
                sentences_o.append(outcome)

                next_chars.append(line[i])
                next_chars_group.append(line_group[i])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        num_features = len(chars) + len(chars_group) + 1
        print('num features: {}'.format(num_features))
        print('MaxLen: ', maxlen)
        X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)
        y_o = np.zeros((len(sentences)), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        X[i, t + leftpad, char_indices[c]] = 1
                for g in chars_group:
                    if g == sentence_group[t]:
                        X[i, t + leftpad, len(chars) + char_indices_group[g]] = 1
                X[i, t + leftpad, len(chars) + len(chars_group)] = t + 1
            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 - softness
                else:
                    y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)
            for g in target_chars_group:
                if g == next_chars_group[i]:
                    y_g[i, target_char_indices_group[g]] = 1 - softness
                else:
                    y_g[i, target_char_indices_group[g]] = softness / (len(target_chars_group) - 1)
            y_o[i] = sentences_o[i]

        for fold in range(folds):
            model = TrainCFR._build_model(maxlen, num_features, target_chars, target_chars_group, use_old_model)
            checkpoint_name = create_checkpoints_path(log_name, models_folder, fold, 'CFR')
            TrainCFR._train_model(model, checkpoint_name, X, y_a, y_o, y_g)
