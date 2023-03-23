import argparse
import tensorflow as tf

from evaluation.evaluator import Evaluator
from commons import utils, shared_variables as shared
from training.train_cf import TrainCF
from training.train_cfr import TrainCFR


class ExperimentRunner:
    _log_names = [
        'Synthetic log labelled.xes',
        'sepsis_cases_1.xes',
        'sepsis_cases_2.xes',
        'sepsis_cases_4.xes',
        'Production.xes'
    ]

    def __init__(self, model, port, python_port, train, evaluate):
        self._model = model
        self._port = port
        self._python_port = python_port
        self._train = train
        self._evaluate = evaluate

        self._evaluator = Evaluator()

        print(args.port, python_port)
        print(self._model)

    def _run_single_experiment(self, log_name):
        xes_log_path = shared.log_folder / log_name
        log_name = utils.encode_log(xes_log_path)

        print('log_name:', log_name)
        print('train:', self._train)
        print('evaluate:', self._evaluate)

        if self._train:
            TrainCF.train(log_name, self._model)
            TrainCFR.train(log_name, self._model)
        if self._evaluate:
            self._evaluator.evaluate_all(log_name, self._model)

    def run_experiments(self, input_log_name):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                                          allow_soft_placement=True)
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

        if input_log_name is not None:
            self._run_single_experiment(input_log_name)
        else:
            for log_name in self._log_names:
                self._run_single_experiment(log_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--model', default="keras_trans", help='choose among ["NN", "custom_trans", "keras_trans"]')
    parser.add_argument('--port', type=int, default=25333, help='communication port (python port = port + 1)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', default=False, action='store_true', help='train without evaluating')
    group.add_argument('--evaluate', default=False, action='store_true', help='evaluate without training')
    group.add_argument('--full_run', default=True, action='store_true', help='train and evaluate model')

    args = parser.parse_args()

    if args.full_run:
        args.train = True
        args.evaluate = True

    ExperimentRunner(model=args.model,
                     port=args.port,
                     python_port=args.port + 1,
                     train=args.train,
                     evaluate=args.evaluate).run_experiments(input_log_name=args.log)
