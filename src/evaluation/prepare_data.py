"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import csv
import re
from datetime import datetime
import numpy as np
import pm4py
import pandas as pd

from src.commons import utils, shared_variables as shared


def prepare_testing_data(eventlog):
    csvfile = open(shared.log_folder / (eventlog + '.csv'), 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers

    lines_id = []
    lines = []  # list of all the activity sequences
    lines_group = []  # list of all the resource sequences
    outcomes = []  # outcome of each activity sequence (i.e. each case)

    lastcase = ''
    line = ''
    line_group = ''
    outcome = ''
    first_line = True
    numlines = 0

    for row in spamreader:  # the rows are "CaseID,ActivityID,CompleteTimestamp,Resource,Label"
        if row[0] != lastcase:  # 'lastcase' is to save the last executed case for the loop
            lastcase = row[0]
            outcome = row[4]
            if not first_line:  # here we actually add the sequences to the lists
                lines.append(line)
                lines_group.append(line_group)
            lines_id.append(lastcase)
            outcomes.append(outcome)
            line = ''
            line_group = ''
            numlines += 1
        line += utils.get_unicode_from_int(row[1])
        line_group += utils.get_unicode_from_int(row[3])
        first_line = False

    # add last case
    lines.append(line)
    lines_group.append(line_group)
    numlines += 1

    elems_per_fold = int(round(numlines / 3))

    fold1and2lines = lines[: 2*elems_per_fold]
    # fold1and2lines = map(lambda x: x + '!', fold1and2lines)
    # maxlen = max(map(lambda x: len(x), fold1and2lines))
    fold1and2lines = [x + '!' for x in fold1and2lines]
    maxlen = max([len(x) for x in fold1and2lines])
    chars = list(map(lambda x: set(x), fold1and2lines))
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    if '!' in chars:
        chars.remove('!')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

    fold1and2lines_group = lines_group[: 2*elems_per_fold]
    # fold1and2lines_group = map(lambda x: x + '!', fold1and2lines_group)
    chars_group = list(map(lambda x: set(x), fold1and2lines_group))
    chars_group = list(set().union(*chars_group))
    chars_group.sort()
    target_chars_group = copy.copy(chars_group)
    # chars_group.remove('!')
    char_indices_group = dict((c, i) for i, c in enumerate(chars_group))
    target_char_indices_group = dict((c, i) for i, c in enumerate(target_chars_group))
    target_indices_char_group = dict((i, c) for i, c in enumerate(target_chars_group))

    # we only need the third fold, because first two were used for training
    lines = lines[2*elems_per_fold:]
    lines_id = lines_id[2*elems_per_fold:]
    lines_group = lines_group[2*elems_per_fold:]
    lines_o = outcomes[2*elems_per_fold:]

    # set parameters
    predict_size = maxlen

    return lines, lines_id, lines_group, lines_o, maxlen, chars, chars_group, char_indices, char_indices_group, \
        predict_size, target_indices_char, target_indices_char_group, target_char_indices, target_char_indices_group


# selects traces verified by a petri net model
def select_petrinet_verified_traces(log_name, lines, lines_id, lines_group, lines_o, path_to_pn_model_file):
    # select only lines with formula verified
    lines_v = []
    lines_id_v = []
    lines_group_v = []
    lines_o_v = []

    for line, line_id, line_group, outcome in zip(lines, lines_id, lines_group, lines_o):
        if get_pn_fitness(path_to_pn_model_file, line_id, line, line_group) >= shared.log_settings[log_name]['th_compliance']:
            lines_v.append(line)
            lines_id_v.append(line_id)
            lines_group_v.append(line_group)
            lines_o_v.append(outcome)

    return lines_v, lines_id_v, lines_group_v, lines_o_v


def get_pn_fitness(pn_file: str, trace_id: str, activities, groups=None):
    if groups is not None:
        log = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'concept:name', 'org:group'])
    else:
        log = pd.DataFrame(columns=['case:concept:name', 'time:timestamp', 'concept:name'])

    for i in range(len(activities)):
        act_name = shared.act_encoding[utils.get_int_from_unicode(activities[i])]
        # Adding time just for avoiding exception (the log dataframe needs it mandatory)
        row = [trace_id, datetime.utcnow().isoformat(), act_name]

        if groups is not None:
            res_name = shared.res_encoding[utils.get_int_from_unicode(groups[i])]
            row.append(res_name)

        # Appending event to the dataframe
        log.loc[len(log)] = row

    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

    net, initial_marking, final_marking = pm4py.read_pnml(pn_file)
    alignment = pm4py.conformance_diagnostics_alignments(log, net, initial_marking, final_marking)[0]
    return alignment['fitness']


# define helper functions
# this one encodes the current sentence into the onehot encoding
def encode(sentence, maxlen, chars, char_indices):
    num_features = len(chars) + 1
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        x[0, t + leftpad, len(chars)] = t + 1
    return x


# define helper functions
# this one encodes the current sentence into the onehot encoding
def encode_with_group(sentence, sentence_group, maxlen, chars, chars_group, char_indices, char_indices_group):
    num_features = len(chars) + len(chars_group) + 1
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    for t, char in enumerate(sentence):
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        for g in chars_group:
            if g == sentence_group[t]:
                x[0, t + leftpad, len(char_indices) + char_indices_group[g]] = 1
        x[0, t + leftpad, len(chars) + len(chars_group)] = t + 1
    return x


# find repetitions
def repetitions(s):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(s):
        yield match.group(1), len(match.group(0)) / len(match.group(1))


def reduce_loop_probability(trace):
    tmp = dict()

    # num_repetitions = number of consequent repetitions of loop inside trace
    for loop, num_repetitions in repetitions(trace):
        if trace.endswith(loop):
            loop_start_symbol = loop[0]
            reduction_factor = 1 / np.math.exp(num_repetitions)

            if loop_start_symbol in tmp:
                if reduction_factor > tmp[loop_start_symbol]:
                    tmp[loop_start_symbol] = reduction_factor
            else:
                tmp[loop_start_symbol] = reduction_factor

    for loop_start_symbol, reduction_factor in tmp.items():
        yield loop_start_symbol, reduction_factor


def get_symbol(prefix, predictions, target_indices_char, target_char_indices, reduce_loop_prob=True, ith_best=0):
    if reduce_loop_prob:
        for symbol_where_loop_starts, reduction_factor in reduce_loop_probability(prefix):
            # Reducing probability of the first symbol of detected loop (if any) for preventing endless traces
            symbol_idx = target_char_indices[symbol_where_loop_starts]
            predictions[symbol_idx] *= reduction_factor

    pred_idx = np.argsort(predictions)[len(predictions) - ith_best - 1]
    return target_indices_char[pred_idx]


def get_group_symbol(predictions, target_indices_char_group, ith_best=0):
    group_pred_idx = np.argsort(predictions)[len(predictions) - ith_best - 1]
    return target_indices_char_group[group_pred_idx]
