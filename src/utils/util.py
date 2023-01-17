import pandas as pd


def encode_activities_and_resources(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    act_dict = dict()
    res_dict = dict()

    for index, event in df.iterrows():
        act_name = event['ActivityID'] if "ActivityID" in df.columns else None
        res_name = event['Resource'] if "Resource" in df.columns else None

        # Find encoding for current event name and resource
        act_encoding = None
        if act_name in act_dict.values():
            act_encoding = [k for k, v in act_dict.items() if v == act_name][0]
        else:
            act_encoding = len(act_dict)
            act_dict[act_encoding] = act_name

        res_encoding = None
        if res_name in res_dict.values():
            res_encoding = [k for k, v in res_dict.items() if v == res_name][0]
        else:
            res_encoding = len(res_dict)
            res_dict[res_encoding] = res_name

        # Encode current event name and resource
        df.at[index, 'ActivityID'] = str(act_encoding)
        df.at[index, 'Resource'] = str(res_encoding)

    return df, act_dict, res_dict
