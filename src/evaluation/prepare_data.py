"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import copy
import re
from pathlib import Path

import numpy as np
import pm4py
import pandas as pd

from src.commons.log_utils import LogData


def prepare_testing_data(act_name_key: str, res_name_key: str, training_traces: pd.DataFrame):
    """
    Get all possible symbols for activities and resources and annotate them with integers.
    """
    act_chars = list(training_traces[act_name_key].unique())
    act_chars.sort()

    target_act_chars = copy.copy(act_chars)
    target_act_chars.append('!')
    target_act_chars.sort()

    act_to_int = dict((c, i) for i, c in enumerate(act_chars))
    target_act_to_int = dict((c, i) for i, c in enumerate(target_act_chars))
    target_int_to_act = dict((i, c) for i, c in enumerate(target_act_chars))

    res_chars = list(training_traces[res_name_key].unique())
    res_chars.sort()

    target_res_chars = copy.copy(res_chars)

    res_to_int = dict((c, i) for i, c in enumerate(res_chars))
    target_res_to_int = dict((c, i) for i, c in enumerate(target_res_chars))
    target_int_to_res = dict((i, c) for i, c in enumerate(target_res_chars))

    return act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res


# selects traces verified by a petri net model
def select_petrinet_compliant_traces(log_data: LogData, traces: pd.DataFrame, path_to_pn_model_file: Path):
    compliant_trace_ids = []
    for trace_id, fitness in get_pn_fitness(path_to_pn_model_file, traces, log_data).items():
        if fitness >= log_data.compliance_th:
            compliant_trace_ids.append(trace_id)

    compliant_traces = traces[traces[log_data.case_name_key].isin(compliant_trace_ids)]
    return compliant_traces


def get_pn_fitness(pn_file: Path, log: pd.DataFrame, log_data: LogData) -> dict[str: float]:
    # Decode traces for feeding them to the Petri net
    dec_log = log.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
        log_data.res_name_key: log_data.res_enc_mapping
    })

    net, initial_marking, final_marking = pm4py.read_pnml(str(pn_file))
    alignments = pm4py.conformance_diagnostics_alignments(dec_log, net, initial_marking, final_marking,
                                                          activity_key=log_data.act_name_key,
                                                          timestamp_key=log_data.timestamp_key,
                                                          case_id_key=log_data.case_name_key)

    trace_ids = list(log[log_data.case_name_key].unique())
    trace_fitnesses = [a['fitness'] for a in alignments]
    trace_ids_with_fitness = dict(zip(trace_ids, trace_fitnesses))
    return trace_ids_with_fitness


# define helper functions
# this one encodes the current sentence into the onehot encoding
def encode(sentence: str, maxlen: int, char_indices: dict[str, int]) -> np.ndarray:
    chars = list(char_indices.keys())
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
def encode_with_group(sentence: str, sentence_group: str, maxlen: int, char_indices: dict[str, int],
                      char_indices_group: dict[str, int]) -> np.ndarray:
    chars = list(char_indices.keys())
    chars_group = list(char_indices_group.keys())
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
