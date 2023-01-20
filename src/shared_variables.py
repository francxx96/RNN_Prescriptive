"""
This file was created in order to bring
common variables and functions into one file to make
code more clear

"""
import glob
import os
from pathlib import Path
import pm4py
from utils.util import encode_activities_and_resources
import pandas as pd

ascii_offset = 161
beam_size = 3

project_folder = Path.cwd().parents[0]
outputs_folder = project_folder / 'output_files'
resources_folder = project_folder / 'resources'

data_folder = resources_folder / 'data'
declare_models_folder = resources_folder / 'declare_models_xml'
pn_models_folder = resources_folder / 'pn_models'

act_encoding = dict()
res_encoding = dict()

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

'''
def extract_declare_model_filename(log_name):
    return str(declare_models_folder / (log_name + ".xml"))
'''


def extract_petrinet_filename(log_name):
    return str(pn_models_folder / (log_name + '.pnml'))


def encode_log(log_path: Path) -> str:
    log_filename = log_path.name

    print(f'Encoding {log_filename} ...')
    if log_filename.endswith('.xes') or log_filename.endswith('.xes.gz'):
        xes_log = pm4py.read_xes(str(log_path))

        csv_df = xes_log.rename(columns={'concept:name': 'ActivityID', 'case:concept:name': 'CaseID',
                                              'org:group': 'Resource', 'time:timestamp': 'CompleteTimestamp'})
        # csv_df = csv_df.drop(['lifecycle:transition', 'case:trace:type'], axis=1)
        csv_df = csv_df[['CaseID', 'ActivityID', 'CompleteTimestamp', 'Resource']]
        csv_df['CompleteTimestamp'] = pd.to_datetime(csv_df['CompleteTimestamp'], utc=True).dt.tz_localize(None)
        csv_df['CompleteTimestamp'] = csv_df['CompleteTimestamp'].dt.round('s')

    # elif log_filename.endswith('.csv'):
    #    csv_df = pd.read_csv(filepath_or_buffer=str(log_path), sep=',')

    else:
        raise RuntimeError(f"Extension of {log_filename} must be '.xes' or '.xes.gz'!")

    global act_encoding
    global res_encoding
    csv_df, act_encoding, res_encoding = encode_activities_and_resources(csv_df)

    if log_filename.endswith('.xes'):
        log_name = log_path.stem
    else:   # endswith '.xes.gz'
        log_name = log_path.with_suffix("").stem

    enc_log_path = log_path.parent / (log_name + '.csv')
    csv_df.to_csv(str(enc_log_path), index=False)

    return enc_log_path.stem


log_settings = {
    'Data-flow log': {
        'formula': "",
        'prefix_size_pred_from': 7,
        'prefix_size_pred_to': 8
    },
}
