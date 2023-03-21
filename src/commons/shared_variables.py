"""
This file was created in order to bring
common variables and functions into one file to make
code more clear
"""
from pathlib import Path


ascii_offset = 161
beam_size = 3

data_folder = Path.cwd().parents[0] / 'data'
input_folder = data_folder / 'input'
output_folder = data_folder / 'output'

log_folder = input_folder / 'logs'
pn_folder = input_folder / 'petrinets'

# these dictionaries are populated with name encoding for activities and resources, each name is encoded with an integer
act_encoding = dict()
res_encoding = dict()

epochs = 300
folds = 3
validation_split = 0.2

log_settings = {
    'Synthetic log labelled': {
        'trace_name_key': 'concept:name',
        'label_key': 'label',
        'label_pos_val': 'positive',
        'label_neg_val': 'negative',
        'act_name_key': 'concept:name',
        'res_name_key': 'org:group',
        'time_key': 'time:timestamp',
        'prefix_size_pred_range': [7, 8],
        'th_compliance': 1.0,
        'th_evaluation': 1.0 * 0.9
    },

    'sepsis_cases_1': {
        'trace_name_key': 'Case ID',
        'label_key': 'label',
        'label_pos_val': 'deviant',
        'label_neg_val': 'regular',
        'act_name_key': 'Activity',
        'res_name_key': 'org:group',
        'time_key': 'time:timestamp',
        'prefix_size_pred_range': [10, 12],
        'th_compliance': 0.77,   # 0.62 for complete petrinet, 0.77 for reduced petrinet
        'th_evaluation': 0.77 * 0.9
    },

    'sepsis_cases_2': {
        'trace_name_key': 'Case ID',
        'label_key': 'label',
        'label_pos_val': 'deviant',
        'label_neg_val': 'regular',
        'act_name_key': 'Activity',
        'res_name_key': 'org:group',
        'time_key': 'time:timestamp',
        'prefix_size_pred_range': [10, 12],
        'th_compliance': 0.55,
        'th_evaluation': 0.55 * 0.9
    },

    'sepsis_cases_4': {
        'trace_name_key': 'Case ID',
        'label_key': 'label',
        'label_pos_val': 'deviant',
        'label_neg_val': 'regular',
        'act_name_key': 'Activity',
        'res_name_key': 'org:group',
        'time_key': 'time:timestamp',
        'prefix_size_pred_range': [10, 12],
        'th_compliance': 0.77,
        'th_evaluation': 0.77 * 0.9
    },
}
