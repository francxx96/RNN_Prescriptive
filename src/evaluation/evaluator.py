from .inference_algorithms import baseline_1_cf, baseline_1_cfr, baseline_2_cf, baseline_2_cfr
from shared_variables import folds


class Evaluator:
    def _start_server_and_evaluate(self, evaluation_algorithm, log_name, models_folder, fold):
        evaluation_algorithm.run_experiments(log_name, models_folder, fold)

    def evaluate_all(self, log_name, models_folder):
        for fold in range(folds):
            print('baseline_1 CF')
            self._start_server_and_evaluate(baseline_1_cf, log_name, models_folder, fold)
            print('baseline_2 CF')
            self._start_server_and_evaluate(baseline_2_cf, log_name, models_folder, fold)
            print('baseline_1 CFR')
            self._start_server_and_evaluate(baseline_1_cfr, log_name, models_folder, fold)
            print('baseline_2 CFR')
            self._start_server_and_evaluate(baseline_2_cfr, log_name, models_folder, fold)

