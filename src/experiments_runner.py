import argparse

import tensorflow as tf
import shared_variables

from evaluation.evaluator import Evaluator
from training.train_cf import TrainCF
from training.train_cfr import TrainCFR
from training.train_cfrt import TrainCFRT
from utils.convert_xes_to_csv import convert_xes_to_csv


class ExperimentRunner:
    _log_names = [
        'Data-flow log.csv'
    ]

    def __init__(self, use_old_model, use_time, port, python_port, train, evaluate):
        self._use_old_model = use_old_model
        self._use_time = use_time
        self._port = port
        self._python_port = python_port
        self._train = train
        self._evaluate = evaluate

        if use_old_model:
            self._models_folder = 'old_model'
        else:
            self._models_folder = 'new_model'

        self._evaluator = Evaluator()
        print(self._models_folder)

    def _run_single_experiment(self, log_name):
        if log_name.endswith('.xes') or log_name.endswith('.xes.gz'):
            print(f'Converting {log_name} to csv...')
            xes_log_path = shared_variables.data_folder.joinpath(log_name)
            log_name = convert_xes_to_csv(xes_log_path)

        print('log_name:', log_name)
        print('use_time:', self._use_time)
        print('train:', self._train)
        print('evaluate:', self._evaluate)

        if self._use_time:
            if self._train:
                TrainCFRT.train(log_name, self._models_folder, self._use_old_model)
            if self._evaluate:
                self._evaluator.evaluate_time(log_name, self._models_folder)
        else:
            if self._train:
                TrainCF.train(log_name, self._models_folder, self._use_old_model)
                TrainCFR.train(log_name, self._models_folder, self._use_old_model)
            if self._evaluate:
                self._evaluator.evaluate_all(log_name, self._models_folder)

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
    ExperimentRunner(use_old_model=True,
                     use_time=False,
                     port=25333,
                     python_port=25334,
                     train=True,
                     evaluate=True).run_experiments(None)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='input log')
    parser.add_argument('--use_old_model', action='store_true', help='use old model')
    parser.add_argument('--use_time', action='store_true', help='use time')
    parser.add_argument('--port', type=int, default=25333, help='communication port (python port = port + 1)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='train without evaluating')
    group.add_argument('--evaluate', action='store_true', help='evaluate without training')
    group.add_argument('--full_run', action='store_true', help='train and evaluate model')

    args = parser.parse_args()

    if args.full_run:
        args.train = True
        args.evaluate = True

    python_port = args.port + 1

    print(args.port, python_port)
    
    ExperimentRunner(use_old_model=args.use_old_model,
                     use_time=args.use_time,
                     port=args.port,
                     python_port=python_port,
                     train=args.train,
                     evaluate=args.evaluate).run_experiments(input_log_name=args.log)
    '''