import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py

import src.commons.shared_variables as shared


class LogName(Enum):
    SYNTH = 'Synthetic log labelled'
    SEPSIS1 = 'sepsis_cases_1'
    SEPSIS2 = 'sepsis_cases_2'
    SEPSIS4 = 'sepsis_cases_4'
    PROD = 'Production'


class LogExt(Enum):
    CSV = '.csv'
    XES = '.xes'
    XES_GZ = '.xes.gz'


class LogData:
    log: pd.DataFrame
    log_name: LogName
    log_ext: LogExt
    training_trace_ids = [str]
    evaluation_trace_ids = [str]

    # Gathered from encoding
    act_enc_mapping: {str, str}
    res_enc_mapping: {str, str}

    # Gathered from manual log analisys
    case_name_key: str
    act_name_key: str
    res_name_key: str
    timestamp_key: str
    label_name_key: str
    label_pos_val: str
    label_neg_val: str
    compliance_th: float
    evaluation_th: float
    evaluation_prefix_start: int
    evaluation_prefix_end: int

    def __init__(self, log_path: Path):
        file_name = log_path.name
        if file_name.endswith('.xes') or file_name.endswith('.xes.gz'):
            if file_name.endswith('.xes'):
                self.log_name = LogName(log_path.stem)
                self.log_ext = LogExt.XES
            else:  # endswith '.xes.gz'
                self.log_name = LogName(log_path.with_suffix("").stem)
                self.log_ext = LogExt.XES_GZ

            self._set_log_keys_and_ths()
            self.log = pm4py.read_xes(str(log_path))[
                [self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            ]
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('.csv'):
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path, sep=';',
                usecols=[self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            )
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        else:
            raise RuntimeError(f"Extension of {file_name} must be in ['.xes', '.xes.gz', '.csv'].")

        # Use last fold for evaluation, remaining ones for training
        trace_ids = self.log[self.case_name_key].unique().tolist()
        elements_per_fold = round(len(trace_ids) / shared.folds)
        self.training_trace_ids = trace_ids[:-elements_per_fold]
        self.evaluation_trace_ids = trace_ids[-elements_per_fold:]

    def encode_log(self):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx + shared.ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)

        res_set = list(self.log[self.res_name_key].unique())
        self.res_enc_mapping = dict((chr(idx + shared.ascii_offset), elem) for idx, elem in enumerate(res_set))
        self.log.replace(to_replace={self.res_name_key: {v: k for k, v in self.res_enc_mapping.items()}}, inplace=True)

        self.log.replace(to_replace={self.label_name_key: {self.label_pos_val: '1', self.label_neg_val: '0'}}, inplace=True)

    def _set_log_keys_and_ths(self):
        # In case of log saved with XES format, case attributes must have the 'case:' prefix
        addit = '' if self.log_ext == LogExt.CSV else 'case:'

        if self.log_name == LogName.SYNTH:
            self.case_name_key = addit+'concept:name'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'positive'
            self.label_neg_val = 'negative'
            self.act_name_key = 'concept:name'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'
            self.compliance_th = 1.0
            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 7
            self.evaluation_prefix_end = 7

        elif self.log_name == LogName.SEPSIS1 \
                or self.log_name == LogName.SEPSIS2 \
                or self.log_name == LogName.SEPSIS4:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'

            if self.log_name == LogName.SEPSIS1:
                self.compliance_th = 0.77   # 0.62 for complete petrinet, 0.77 for reduced petrinet
            elif self.log_name == LogName.SEPSIS2:
                self.compliance_th = 0.55
            else:   # log_name == LogName.SEPSIS4
                self.compliance_th = 0.77

            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 10
            self.evaluation_prefix_end = 10

        elif self.log_name == LogName.PROD:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit + 'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'Resource'
            self.timestamp_key = 'Complete Timestamp'
            self.compliance_th = 0.86
            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 7
            self.evaluation_prefix_end = 7

        else:
            raise RuntimeError(f"No settings defined for log: {self.log_name.value}.")
