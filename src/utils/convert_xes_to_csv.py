import pm4py
import pandas as pd
from pathlib import Path


def convert_xes_to_csv(xes_log_path: Path) -> str:
    xes_log = pm4py.read_xes(str(xes_log_path))
    dataframe = pm4py.convert_to_dataframe(xes_log)
    dataframe = dataframe.rename(columns={'concept:name': 'ActivityID', 'case:concept:name': 'CaseID',
                                          'resource': 'Resource', 'time:timestamp': 'CompleteTimestamp'})
    # dataframe = dataframe.drop(['lifecycle:transition', 'case:trace:type'], axis=1)
    dataframe = dataframe[['CaseID', 'ActivityID', 'CompleteTimestamp', 'Resource']]
    dataframe['CompleteTimestamp'] = pd.to_datetime(dataframe['CompleteTimestamp'], utc=True).dt.tz_localize(None)
    dataframe['CompleteTimestamp'] = dataframe['CompleteTimestamp'].dt.round('s')

    csv_log_path = Path()
    if xes_log_path.name.endswith('.xes'):
        csv_log_path = xes_log_path.with_suffix('.csv')
    elif xes_log_path.name.endswith('.xes.gz'):
        csv_log_path = xes_log_path.with_suffix('').with_suffix('.csv')

    dataframe.to_csv(str(csv_log_path), index=False)

    return csv_log_path.stem

