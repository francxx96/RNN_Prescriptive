from src.evaluation.inference_algorithms import baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr
from src.commons import shared_variables as shared


class Evaluator:
    def evaluate_all(self, log_name, models_folder):
        for fold in range(shared.folds):
            print(f"fold {fold} - baseline CF")
            self._evaluate(baseline_cf, log_name, models_folder, fold)
            print(f"fold {fold} - beamsearch CF")
            self._evaluate(beamsearch_cf, log_name, models_folder, fold)
            print(f"fold {fold} - baseline CFR")
            self._evaluate(baseline_cfr, log_name, models_folder, fold)
            print(f"fold {fold} - beamsearch CFR")
            self._evaluate(beamsearch_cfr, log_name, models_folder, fold)

    def _evaluate(self, evaluation_algorithm, log_name, models_folder, fold):
        evaluation_algorithm.run_experiments(log_name, models_folder, fold)
