import time
from pathlib import Path

from src.commons.log_utils import LogData
from src.commons import shared_variables as shared
from src.commons.utils import extract_petrinet_filename, extract_last_model_checkpoint
from src.evaluation.prepare_data import prepare_testing_data, select_petrinet_compliant_traces
from src.evaluation.inference_algorithms import baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr


def evaluate_all(log_data: LogData, models_folder: str):
    start_time = time.time()

    training_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.training_trace_ids)]
    # +1 accounts of fake '!' symbol added to identify end of trace
    maxlen = max([trace.shape[0] for _, trace in training_traces.groupby(log_data.case_name_key)]) + 1
    predict_size = maxlen

    act_to_int, target_act_to_int, target_int_to_act, res_to_int, target_res_to_int, target_int_to_res \
        = prepare_testing_data(log_data.act_name_key, log_data.res_name_key, training_traces)

    pn_filename = extract_petrinet_filename(log_data.log_name.value)
    # Extract evaluation sequences compliant to the background knowledge
    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    compliant_traces = select_petrinet_compliant_traces(log_data, evaluation_traces, pn_filename)

    print("Compliant traces: " + str(compliant_traces[log_data.case_name_key].nunique())
          + " out of " + str(len(log_data.evaluation_trace_ids)))
    print('Elapsed time:', time.time() - start_time)

    for fold in range(shared.folds):
        for eval_algorithm in [baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr]:
            start_time = time.time()

            algorithm_name = Path(eval_algorithm.__file__).stem
            folder_path = shared.output_folder / models_folder / str(fold) / 'results' / algorithm_name
            if not Path.exists(folder_path):
                Path.mkdir(folder_path, parents=True)
            output_filename = folder_path / f'{log_data.log_name.value}.csv'

            print(f"fold {fold} - {algorithm_name}")
            if eval_algorithm is baseline_cf:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF')
                baseline_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                            target_act_to_int, target_int_to_act, model_filename, output_filename)

            elif eval_algorithm is beamsearch_cf:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CF')
                beamsearch_cf.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                              target_act_to_int, target_int_to_act, model_filename, output_filename,
                                              pn_filename)

            elif eval_algorithm is baseline_cfr:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFR')
                baseline_cfr.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                             target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                             target_int_to_res, model_filename, output_filename)

            elif eval_algorithm is beamsearch_cfr:
                model_filename = extract_last_model_checkpoint(log_data.log_name.value, models_folder, fold, 'CFR')
                beamsearch_cfr.run_experiments(log_data, compliant_traces, maxlen, predict_size, act_to_int,
                                               target_act_to_int, target_int_to_act, res_to_int, target_res_to_int,
                                               target_int_to_res, model_filename, output_filename, pn_filename)
            else:
                raise RuntimeError(f"No model called {eval_algorithm}.")

            print("TIME TO FINISH --- %s seconds ---" % (time.time() - start_time))
