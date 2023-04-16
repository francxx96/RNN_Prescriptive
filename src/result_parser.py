import argparse
import csv
from pathlib import Path

import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

from src.commons import shared_variables as shared
from src.evaluation.inference_algorithms import baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr


class ResultParser:
    class HighlightTypes(Enum):
        ROW_SCORE = 0
        IMPROVEMENT_SCORE = 1

    class ColumnTypes(Enum):
        CF = 0
        R = 1
        O = 2

    _headers = ['BL_CF', 'BL_R', 'BS_CF', 'BS_R']
    _eval_algorithms = [baseline_cf, baseline_cfr, beamsearch_cf, beamsearch_cfr]

    # <editor-fold desc="reference_table">
    _reference_table = np.array(
        [[0.693, 0.742, 0.511, 0.658, 0.804, 0.729, 0.390, 0.765, 0.727, 0.784, 0.827, 0.849, 0.618,
          0.729, 0.727, 0.409, 0.553, 0.314, 0.231, 0.623, 0.000, 0.000],
         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
         [0.643, 0.806, 0.258, 0.629, 0.810, 0.723, 0.695, 0.831, 0.864, 0.761, 0.784, 0.774, 0.645,
          0.749, 0.655, 0.806, 0.735, 0.506, 0.466, 0.802, 0.000, 0.000],
         [0.674, 0.682, 0.803, 0.793, 0.800, 0.605, 0.700, 0.741, 0.579, 0.583, 0.608, 0.566, 0.677,
          0.574, 0.642, 0.644, 0.642, 0.308, 0.377, 0.803, 0.000, 0.000],
         [0.649, 0.742, 0.682, 0.682, 0.803, 0.719, 0.479, 0.769, 0.909, 0.784, 0.884, 0.556, 0.616,
          0.729, 0.726, 0.413, 0.746, 0.674, 0.874, 0.732, 0.000, 0.000],
         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
          0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
         [0.665, 0.806, 0.536, 0.662, 0.813, 0.723, 0.720, 0.846, 0.855, 0.723, 0.884, 0.620, 0.639,
          0.750, 0.691, 0.806, 0.707, 0.819, 0.825, 0.789, 0.000, 0.000],
         [0.640, 0.682, 0.760, 0.826, 0.800, 0.605, 0.717, 0.761, 0.579, 0.571, 0.627, 0.449, 0.684,
          0.574, 0.650, 0.643, 0.561, 0.514, 0.623, 0.799, 0.000, 0.000],
         [0.646, 0.803, 0.695, 0.667, 0.780, 0.831, 0.802, 0.849, 0.864, 0.838, 0.884, 0.000, 0.601,
          0.754, 0.718, 0.846, 0.697, 0.735, 0.870, 0.811, 0.000, 0.000],
         [0.649, 0.679, 0.763, 0.833, 0.790, 0.824, 0.792, 0.761, 0.575, 0.738, 0.704, 0.000, 0.785,
          0.758, 0.763, 0.679, 0.687, 0.742, 0.875, 0.815, 0.000, 0.000]]).T

    # </editor-fold>

    _column_colors = {
        ColumnTypes.CF: 'E0E0E0',
        ColumnTypes.R: 'C0C0C0',
        ColumnTypes.O: 'A0A0A0'
    }

    def __init__(self, log_filenames):
        self._log_names = []

        for log_filename in log_filenames:
            log_path = shared.log_folder / log_filename
            if log_filename.endswith('.xes') or log_filename.endswith('.csv'):
                log_name = log_path.stem
            elif log_filename.endswith('.xes.gz'):
                log_name = log_path.with_suffix("").stem
            else:
                raise RuntimeError(f"Extension of {log_filename} must be in ['.xes', '.xes.gz', '.csv'].")

            self._log_names.append(log_name)

    @staticmethod
    def _parse_log(filepath, resource_prediction=False):
        label_1 = 'Damerau-Levenshtein'
        label_2 = 'Damerau-Levenshtein Resource'
        label_3 = 'Outcome diff.'

        scores = [[], [], []]

        with open(filepath, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='|')
            csv_headers = next(csv_reader)

            for row in csv_reader:
                if row:
                    score_1 = float(row[csv_headers.index(label_1)])
                    scores[0].append(score_1)

                    score_2 = float(row[csv_headers.index(label_2)]) if resource_prediction else 0.0
                    scores[1].append(score_2)

                    score_3 = float(row[csv_headers.index(label_3)])
                    scores[2].append(score_3)

        scores = np.mean(np.array(scores), -1)
        return scores

    def _populate_table(self, table, scores, log_name, eval_algorithm):
        row = self._log_names.index(log_name)
        column = self._eval_algorithms.index(eval_algorithm) * 3

        table[row, column] = scores[0]
        table[row, column + 1] = scores[1]
        table[row, column + 2] = scores[2]

    @staticmethod
    def _print_latex_table_header():

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Table header/footer works only with the following packages imported in the LaTeX document
        #   \usepackage[table,xcdraw]{xcolor}
        #   \usepackage{graphicx}
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        print('\\begin{table}[!hbt]')
        print('\\resizebox{\\textwidth}{!}{')
        print('\\centering')
        print('\\begin{tabular}{|l||c|c|c|c|c|c||c|c|c|c|c|c||}')
        print('\\hline')
        print('\\textit{KB $\\rightarrow$} & '
              '\\multicolumn{6}{c||}{\\textbf{baseline}} & '
              '\\multicolumn{6}{c||}{\\textbf{beamsearch}} \\\\')
        print('\\hline')
        print('\\textit{Encoding $\\rightarrow$} & '
              '\\multicolumn{3}{c|}{\\textbf{CF}} & '
              '\\multicolumn{3}{c||}{\\textbf{CF+R}} & '
              '\\multicolumn{3}{c|}{\\textbf{CF}} & '
              '\\multicolumn{3}{c||}{\\textbf{CF+R}} \\\\')
        print('\\hline')
        print('\\textit{Predicand $\\rightarrow$} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{O} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{O} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{O} & '
              '\\textbf{CF} & \\textbf{R} & \\textbf{O} '
              '\\\\')
        print('\\hline\\hline')

    @staticmethod
    def _print_latex_table_footer(table_caption, table_label):
        print('\\hline')
        print('\\end{tabular}')
        print('}')
        print('\\caption{' + table_caption + '}')
        print('\\label{' + table_label + '}')
        print('\\end{table}')

    @staticmethod
    def _print_score(score, std, reference_score, highlight_type, column_type):
        column_color = ResultParser._column_colors[column_type]

        if highlight_type == ResultParser.HighlightTypes.ROW_SCORE:
            if score == reference_score:
                print('\\cellcolor[HTML]{%s}\\textbf{%.0f$\\pm$%.0f}' % (column_color, 100*score, 100*std), end=' '),
            else:
                print('%.0f$\\pm$%.0f' % (100*score, 100*std), end=' '),
        elif highlight_type == ResultParser.HighlightTypes.IMPROVEMENT_SCORE:
            if score > 0:
                print('\\cellcolor[HTML]{%s}\\textbf{%.0f$\\pm$%.0f}' % (column_color, 100*score, 100*std), end=' '),
            else:
                print('%.0f$\\pm$%.0f' % (100*score, 100*std), end=' '),
        else:
            print('%.0f$\\pm$%.0f' % (100*score, 100*std), end=' '),

    def _print_latex_table(self, populated_table, std_table, highlight_type, table_caption, table_label):
        cf_maximums = np.max(populated_table[:, 0::3], 1)
        r_maximums = np.max(populated_table[:, 1::3], 1)
        o_maximums = np.max(populated_table[:, 2::3], 1)

        self._print_latex_table_header()
        for i, log_name in enumerate(self._log_names):
            print(log_name.replace('_', '\_') + ' & ', end=' '),
            for j, score in enumerate(populated_table[i]):
                if j % 3 == 0:
                    self._print_score(score, std_table[i][j], cf_maximums[i], highlight_type, ResultParser.ColumnTypes.CF)
                elif j % 3 == 1:
                    self._print_score(score, std_table[i][j], r_maximums[i], highlight_type, ResultParser.ColumnTypes.R)
                else:
                    self._print_score(score, std_table[i][j], o_maximums[i], highlight_type, ResultParser.ColumnTypes.O)

                if j != populated_table.shape[1] - 1:
                    print('&', end=' '),
                else:
                    print('\\\\')

        self._print_latex_table_footer(table_caption, table_label)

    def _show_comparison_image(self, target_table, reference_table):
        improvement_percentage = (1.0 * np.count_nonzero(target_table > reference_table) / np.count_nonzero(
            target_table)) * 100.0

        populated_indexes = np.where(target_table > 0)
        mean_improvement = float(np.mean(target_table[populated_indexes] - reference_table[populated_indexes]))
        sum_improvement = float(np.sum(target_table[populated_indexes] - reference_table[populated_indexes]))

        plt.subplots(1, 2)
        plt.subplot(1, 2, 1)
        plt.title('better performance (%.2f' % improvement_percentage + '%)')
        binary_image = np.zeros(target_table.shape)
        binary_image[target_table == 0] = -1.0
        binary_image[target_table > reference_table] = 1.0
        plt.imshow(binary_image, cmap='hot')
        plt.yticks(range(len(self._log_names)), self._log_names)
        plt.xticks(range(len(self._headers)), self._headers, rotation=90)

        plt.subplot(1, 2, 2)
        plt.title('comparison (mean:%.2f, sum:%.2f' % (mean_improvement, sum_improvement) + ')')
        gradient_image = target_table - reference_table
        gradient_image[target_table == 0] = 0
        plt.imshow(gradient_image, vmin=-1, vmax=1,
                   cmap=plt.cm.seismic)
        plt.yticks(range(len(self._log_names)), self._log_names)
        plt.xticks(range(len(self._headers)), self._headers, rotation=90)
        plt.tight_layout()
        plt.show()

    def _load_table(self, folderpath):
        if folderpath == 'reference':
            return self._reference_table[sorted(self._log_names.index(i) for i in self._log_names)]

        elif folderpath == 'zeros':
            return np.zeros((len(self._log_names), len(self._eval_algorithms) * 3)), \
                   np.zeros((len(self._log_names), len(self._eval_algorithms) * 3))

        else:
            table_folds = []
            for fold in range(shared.folds):
                fold_table = np.zeros((len(self._log_names), len(self._eval_algorithms) * 3))

                for log_name in self._log_names:
                    for alg in self._eval_algorithms:
                        filepath = folderpath / str(fold) / 'results' / Path(alg.__file__).stem / f"{log_name}.csv"
                        scores = self._parse_log(filepath, alg in [baseline_cfr, beamsearch_cfr])
                        self._populate_table(fold_table, scores, log_name, alg)
                table_folds.append(fold_table)
            populated_table = np.mean(table_folds, axis=0)
            std_table = np.std(table_folds, axis=0)
            return populated_table, std_table

    def compare_results(self, target, reference='zeros', highlight_type=HighlightTypes.ROW_SCORE, table_caption='',
                        table_label=''):
        target_table, std_target = self._load_table(target)
        reference_table, std_ref = self._load_table(reference)

        # self._show_comparison_image(target_table, reference_table)
        self._print_latex_table(target_table - reference_table, std_target - std_ref, highlight_type, table_caption,
                                table_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input logs')
    parser.add_argument('--target_model', default="keras_trans", help='choose among ["LSTM", "custom_trans", "keras_trans"]')
    parser.add_argument('--reference_model', default='zeros', help='reference model name')
    args = parser.parse_args()

    logs = [args.log.strip()] if args.log else shared.log_list

    result_parser = ResultParser(logs)

    models_dict = {
        args.target_model: shared.output_folder / args.target_model,
        'reference': 'reference',
        'zeros': 'zeros'
    }

    caption = f'pn_checker,{args.target_model}'.replace('_', '\_')
    label = f'tb:{args.target_model}'.replace('_', '\_')

    result_parser.compare_results(models_dict[args.target_model], reference=models_dict[args.reference_model],
                                  table_caption=caption, table_label=label)
