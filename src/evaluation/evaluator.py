from src.evaluation.inference_algorithms import baseline_1_cf, baseline_1_cfr, baseline_2_cf, baseline_2_cfr
from src.commons import shared_variables as shared


class Evaluator:
    def evaluate_all(self, log_name, models_folder):
        for fold in range(shared.folds):
            print(f"fold {fold} - baseline_1 CF")
            self._evaluate(baseline_1_cf, log_name, models_folder, fold)
            print(f"fold {fold} - baseline_2 CF")
            self._evaluate(baseline_2_cf, log_name, models_folder, fold)
            print(f"fold {fold} - baseline_1 CFR")
            self._evaluate(baseline_1_cfr, log_name, models_folder, fold)
            print(f"fold {fold} - baseline_2 CFR")
            self._evaluate(baseline_2_cfr, log_name, models_folder, fold)

    def _evaluate(self, evaluation_algorithm, log_name, models_folder, fold):
        evaluation_algorithm.run_experiments(log_name, models_folder, fold)
