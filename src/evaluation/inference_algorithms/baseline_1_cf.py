"""
this script takes as input the LSTM or RNN weights found by train.py
change the path in the shared_variables.py to point to the h5 file
with LSTM or RNN weights generated by train.py

Author: Niek Tax
"""

from __future__ import division

import csv
from pathlib import Path
import time
from datetime import timedelta
from jellyfish import damerau_levenshtein_distance
import distance
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import metrics

import shared_variables
from evaluation.prepare_data_resource import select_declare_verified_traces, prepare_testing_data


def run_experiments(log_name, models_folder, fold):
    model_filename = shared_variables.extract_last_model_checkpoint(log_name, models_folder, fold, 'CF')
    declare_model_filename = shared_variables.extract_declare_model_filename(log_name)

    log_settings_dictionary = shared_variables.log_settings[log_name]
    prefix_size_pred_from = log_settings_dictionary['prefix_size_pred_from']
    prefix_size_pred_to = log_settings_dictionary['prefix_size_pred_to']

    start_time = time.time()

    # prepare the data N.B. maxlen == predict_size
    lines, lines_id, lines_group, lines_t, lines_t2, lines_t3, lines_t4, maxlen, chars, chars_group, char_indices, \
        char_indices_group, divisor, divisor2, divisor3, predict_size, target_indices_char, target_indices_char_group, \
        target_char_indices, target_char_indices_group = prepare_testing_data(log_name)

    # load model, set this to the model generated by train.py
    model = load_model(model_filename)

    # define helper functions
    # this one encodes the current sentence into the onehot encoding
    def encode(sentence, times_enc, times3_enc, maxlen_enc=maxlen):
        num_features = len(chars) + 5
        x = np.zeros((1, maxlen_enc, num_features), dtype=np.float32)
        leftpad = maxlen_enc - len(sentence)
        times2_enc = np.cumsum(times_enc)
        for v, char in enumerate(sentence):
            midnight_enc = times3_enc[v].replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight_enc = times3_enc[v] - midnight_enc
            for c in chars:
                if c == char:
                    x[0, v + leftpad, char_indices[c]] = 1
            x[0, v + leftpad, len(chars)] = v + 1
            x[0, v + leftpad, len(chars) + 1] = times_enc[v] / divisor
            x[0, v + leftpad, len(chars) + 2] = times2_enc[v] / divisor2
            x[0, v + leftpad, len(chars) + 3] = timesincemidnight_enc.seconds / 86400
            x[0, v + leftpad, len(chars) + 4] = times3_enc[v].weekday() / 7
        return x

    # modify to be able to get second-best prediction
    def get_symbol(predictions, jth_best=0):
        j = np.argsort(predictions)[len(predictions) - jth_best - 1]
        return target_indices_char[j]

    one_ahead_gt = []
    one_ahead_pred = []

    folder_path = shared_variables.outputs_folder / models_folder / str(fold) / 'results' / 'baseline'
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    output_filename = folder_path / (log_name + '_CF.csv')

    with open(str(output_filename), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Prefix length", "Ground truth", "Predicted", "Damerau-Levenshtein", "Jaccard",
                             "Ground truth times", "Predicted times", "RMSE", "MAE", "Median AE"])

        curr_time = time.time()
        lines_s, lines_id_s, lines_group_s, lines_t_s, lines_t2_s, lines_t3_s, lines_t4_s \
            = select_declare_verified_traces(log_name, lines, lines_id, lines_group, lines_t, lines_t2, lines_t3,
                                             lines_t4, declare_model_filename, None)

        print("formulas verified: " + str(len(lines_s)) + " out of : " + str(len(lines)))
        print('elapsed_time:', time.time() - curr_time)

        for prefix_size in range(prefix_size_pred_from, prefix_size_pred_to):
            print("prefix size: " + str(prefix_size))
            for line, times, times2, times3 in zip(lines_s, lines_t_s, lines_t2_s, lines_t3_s):
                times.append(0)
                cropped_line = ''.join(line[:prefix_size])
                cropped_times = times[:prefix_size]
                cropped_times3 = times3[:prefix_size]
                if len(times2) < prefix_size:
                    continue  # make no prediction for this case, since this case has ended already
                ground_truth = ''.join(line[prefix_size:prefix_size + predict_size])
                ground_truth_t = times2[prefix_size - 1]
                case_end_time = times2[len(times2) - 1]
                ground_truth_t = case_end_time - ground_truth_t
                predicted = ''
                total_predicted_time = 0

                for i in range(predict_size):
                    enc = encode(cropped_line, cropped_times, cropped_times3)
                    y = model.predict(enc, verbose=0)  # make predictions
                    # split predictions into separate activity and time predictions
                    y_char = y[0][0]
                    y_t = y[1][0][0]
                    prediction = get_symbol(y_char)  # undo one-hot encoding
                    cropped_line += prediction  # add the prediction to the suffix
                    if y_t < 0:
                        y_t = 0
                    cropped_times.append(y_t)

                    # end of case was just predicted, therefore, stop predicting further into the future
                    if prediction == '!':
                        one_ahead_pred.append(total_predicted_time)
                        one_ahead_gt.append(ground_truth_t)
                        # print('! predicted, end case')
                        break
                    y_t = y_t * divisor3
                    cropped_times3.append(cropped_times3[-1] + timedelta(seconds=(int(y_t) if y_t == 0 else y_t)))
                    total_predicted_time = total_predicted_time + y_t
                    predicted += prediction
                output = []
                if len(ground_truth) > 0:
                    output.append(prefix_size)
                    output.append(ground_truth)
                    output.append(predicted)
                    # output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                    dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(
                        len(predicted), len(ground_truth)))
                    if dls < 0:
                        dls = 0
                    output.append(dls)
                    output.append(1 - distance.jaccard(predicted, ground_truth))
                    output.append(ground_truth_t)
                    output.append(total_predicted_time)
                    output.append('')
                    output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                    output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                    spamwriter.writerow(output)
    print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
