import glob
import os
from pathlib import Path

from src.commons import shared_variables as shared
from src.commons.log_utils import LogData


def extract_last_model_checkpoint(log_name: str, models_folder: str, fold: int, model_type: str) -> Path:
    model_filepath = shared.output_folder / models_folder / str(fold) / 'models' / model_type / log_name
    list_of_files = glob.glob(str(model_filepath / '*.h5'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return Path(latest_file)


def extract_petrinet_filename(log_name: str) -> Path:
    return shared.pn_folder / (log_name + '.pnml')


def extract_trace_sequences(log_data: LogData, trace_ids: [str]) -> ([str], [str], [str]):
    """
    Extract activity, resource and output sequences starting from a list of trace ids (i.e. trace_names).
    """
    act_seqs = []  # list of all the activity sequences
    res_seqs = []  # list of all the resource sequences
    outcomes = []  # outcome of each sequence (i.e. each case)

    traces = log_data.log[log_data.log[log_data.case_name_key].isin(trace_ids)]
    for _, trace in traces.groupby(log_data.case_name_key):
        line = ''.join(trace[log_data.act_name_key].tolist())  # sequence of activities for one case
        act_seqs.append(line)

        line_group = ''.join(trace[log_data.res_name_key].tolist())  # sequence of groups for one case
        res_seqs.append(line_group)

        outcome = trace[log_data.label_name_key].iloc[0]
        outcomes.append(outcome)

    return act_seqs, res_seqs, outcomes
