import glob
import os
from pathlib import Path
import pandas as pd
import pm4py

from src.commons import shared_variables as shared


def get_unicode_from_int(ch):
    return chr(int(ch) + shared.ascii_offset)


def get_int_from_unicode(unch):
    return int(ord(unch)) - shared.ascii_offset


def extract_last_model_checkpoint(log_name, models_folder, fold, model_type):
    model_filepath = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    list_of_files = glob.glob(str(model_filepath / '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def extract_petrinet_filename(log_name):
    return str(shared.pn_folder / (log_name + '.pnml'))


def encode_log(log_path: Path) -> str:
    log_filename = log_path.name

    if log_filename.endswith('.xes') or log_filename.endswith('.xes.gz'):
        if log_filename.endswith('.xes'):
            log_name = log_path.stem
        else:   # endswith '.xes.gz'
            log_name = log_path.with_suffix("").stem
    else:
        raise RuntimeError(f"Extension of {log_filename} must be '.xes' or '.xes.gz'!")

    print(f'Encoding {log_filename} ...')
    xes_log = pm4py.read_xes(str(log_path))

    settings = shared.log_settings[log_name]
    case_name = 'case:' + settings['trace_name_key']
    case_label = 'case:' + settings['label_key']
    xes_log[case_label].replace({settings['label_neg_val']: '0', settings['label_pos_val']: '1'}, inplace=True)

    xes_log = xes_log.rename(columns={settings['act_name_key']: 'ActivityID', case_name: 'CaseID',
                                      settings['res_name_key']: 'Resource', settings['time_key']: 'CompleteTimestamp',
                                      case_label: 'Label'})
    # xes_log = xes_log.drop(['lifecycle:transition', 'case:trace:type'], axis=1)
    xes_log = xes_log[['CaseID', 'ActivityID', 'CompleteTimestamp', 'Resource', 'Label']]
    xes_log['CompleteTimestamp'] = pd.to_datetime(xes_log['CompleteTimestamp'], utc=True).dt.tz_localize(None)
    xes_log['CompleteTimestamp'] = xes_log['CompleteTimestamp'].dt.round('s')

    xes_log, shared.act_encoding, shared.res_encoding = encode_activities_and_resources(xes_log)

    enc_log_path = log_path.parent / (log_name + '.csv')
    xes_log.to_csv(str(enc_log_path), index=False)

    return enc_log_path.stem


def encode_activities_and_resources(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    act_dict = dict()
    res_dict = dict()

    for index, event in df.iterrows():
        act_name = event['ActivityID'] if "ActivityID" in df.columns else None
        res_name = event['Resource'] if "Resource" in df.columns else None

        # Find encoding for current event name and resource
        if act_name in act_dict.values():
            act_encoding = [k for k, v in act_dict.items() if v == act_name][0]
        else:
            act_encoding = len(act_dict)
            act_dict[act_encoding] = act_name

        if res_name in res_dict.values():
            res_encoding = [k for k, v in res_dict.items() if v == res_name][0]
        else:
            res_encoding = len(res_dict)
            res_dict[res_encoding] = res_name

        # Encode current event name and resource
        df.at[index, 'ActivityID'] = str(act_encoding)
        df.at[index, 'Resource'] = str(res_encoding)

    return df, act_dict, res_dict
