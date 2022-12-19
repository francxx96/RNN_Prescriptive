import pm4py
from pm4py.objects.log.obj import EventLog, Event
import pandas as pd
from pathlib import Path


def convert_xes_to_csv(xes_log_path: Path) -> str:
    xes_log = pm4py.read_xes(str(xes_log_path))

    act_encoding, res_encoding = encode_activities_and_resources(xes_log)

    dataframe = pm4py.convert_to_dataframe(xes_log)
    dataframe = dataframe.rename(columns={'concept:name': 'ActivityID', 'case:concept:name': 'CaseID',
                                          'org:group': 'Resource', 'time:timestamp': 'CompleteTimestamp'})
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


def encode_activities_and_resources(log: EventLog) -> tuple[dict, dict]:
    act_dict = dict()
    res_dict = dict()

    for trace in log:
        for event in trace:
            act_name = event["concept:name"] if "concept:name" in event.keys() else None
            res_name = event["org:group"] if "org:group" in event.keys() else None

            # Find encoding for current event name and resource
            act_encoding = None
            if (act_name in act_dict.values()):
                act_encoding = [k for k, v in act_dict.items() if v == act_name][0]
            else:
                act_encoding = len(act_dict)
                act_dict[act_encoding] = act_name

            res_encoding = None
            if (res_name in res_dict.values()):
                res_encoding = [k for k, v in res_dict.items() if v == res_name][0]
            else:
                res_encoding = len(res_dict)
                res_dict[res_encoding] = res_name

            # Encode current event name and resource
            event["concept:name"] = str(act_encoding)
            event["org:group"] = str(res_encoding)

    return act_dict, res_dict

