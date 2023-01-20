from .inference_algorithms import \
        baseline_1_cf, baseline_1_cfr, baseline_1_cfrt, baseline_2_cf, baseline_2_cfr, new_method_cfr_ORIG, \
        baseline_2_cf_ORIG, new_method_cfr, baseline_2_cfr_ORIG
from shared_variables import folds


class Evaluator:
    def _start_server_and_evaluate(self, evaluation_algorithm, log_name, models_folder, fold):
        evaluation_algorithm.run_experiments(log_name, models_folder, fold)

    def evaluate_all(self, log_name, models_folder):
        for fold in range(folds):
            print('baseline_1 CF')
            self._start_server_and_evaluate(baseline_1_cf, log_name, models_folder, fold)
            print('baseline_2 CF')
            self._start_server_and_evaluate(baseline_2_cf_ORIG, log_name, models_folder, fold)
            print('baseline_1 CFR')
            self._start_server_and_evaluate(baseline_1_cfr, log_name, models_folder, fold)
            print('baseline_2 CFR')
            self._start_server_and_evaluate(baseline_2_cfr_ORIG, log_name, models_folder, fold)
            # print('new method')
            # self._start_server_and_evaluate(new_method_cfr_ORIG, log_name, models_folder, fold)

    def evaluate_time(self, log_name, models_folder):
        for fold in range(folds):
            print('baseline_1 CFRT')
            self._start_server_and_evaluate(baseline_1_cfrt, log_name, models_folder, fold)
