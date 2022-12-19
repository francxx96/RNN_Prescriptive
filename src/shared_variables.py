"""
This file was created in order to bring
common variables and functions into one file to make
code more clear

"""
import glob
import os
from pathlib import Path

ascii_offset = 161
beam_size = 3
log_file = ''
outputs_folder = Path.cwd() / 'output_files'
data_folder = Path.cwd() / 'data'

declare_models_folder = Path.cwd() / 'declare_models_xml'
epochs = 300
folds = 3
validation_split = 0.2


def get_unicode_from_int(ch):
    return chr(int(ch) + ascii_offset)


def get_int_from_unicode(unch):
    return int(ord(unch)) - ascii_offset


def extract_last_model_checkpoint(log_name, models_folder, fold, model_type):
    model_filepath = outputs_folder / models_folder / str(fold) / 'models' / model_type / log_name
    list_of_files = glob.glob(str(model_filepath / '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def extract_declare_model_filename(log_name):
    return str(declare_models_folder / (log_name + ".xml"))


log_settings = {
    'Data-flow log': {
        'formula': "",
        'prefix_size_pred_from': 3,
        'prefix_size_pred_to': 7
    },
}
