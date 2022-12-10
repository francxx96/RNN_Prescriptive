import pdb

from parsers import *
from mp_checkers.trace.mp_choice import *
from mp_checkers.trace.mp_existence import *
from mp_checkers.trace.mp_negative_relation import *
from mp_checkers.trace.mp_relation import *
import platform


def run_all_mp_checkers_traces(trace, log_name):
    if platform.system() == 'Windows':
        formula_path = '..\\formulas\\' + log_name + '.decl'
    # elif platform.system() == 'Linux':
    else:
        formula_path = '../formulas/' + log_name + '.decl'
    input = parse_decl(formula_path)
    activities = input.activities
    res = True
    for constraint in input.checkers:
        if constraint['key'].startswith(EXISTENCE):
            res = res and mp_existence_checker(trace, True, constraint['attribute'],constraint['condition'][0],
                                               constraint['condition'][1])
        elif constraint['key'].startswith(ABSENCE):
            res = res and mp_absence_checker(trace, True, constraint['attribute'], constraint['condition'][0],
                                             constraint['condition'][1])
        elif constraint['key'].startswith(INIT):
            res = res and mp_init_checker(trace, True, constraint['attribute'], constraint['condition'][0])
        elif constraint['key'].startswith(EXACTLY):
            res = res and mp_exactly_checker(trace, True, constraint['attribute'], constraint['condition'][0],
                                             constraint['condition'][1])
        elif constraint['key'].startswith(CHOICE):
            res = res and mp_choice_checker(trace, True, constraint['attribute'].split(', ')[0],
                                             constraint['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(EXCLUSIVE_CHOICE):
            res = res and mp_exclusive_choice_checker(trace, True, constraint['attribute'].split(', ')[0]
            ['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(RESPONDED_EXISTENCE):
            res = res and mp_responded_existence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                         constraint['attribute'].split(', ')[1],
                                                         constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(RESPONSE):
            res = res and mp_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                    constraint['attribute'].split(', ')[1],
                                                    constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            res = res and mp_alternate_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_RESPONSE):
            res = res and mp_chain_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(PRECEDENCE):
            res = res and mp_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                      constraint['attribute'].split(', ')[1],
                                                      constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_PRECEDENCE):
            res = res and mp_alternate_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_PRECEDENCE):
            res = res and mp_chain_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                            constraint['attribute'].split(', ')[1],
                                                            constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            res = res and mp_not_responded_existence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                   constraint['attribute'].split(', ')[1],
                                                                   constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_RESPONSE):
            res = res and mp_not_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                        constraint['attribute'].split(', ')[1],
                                                        constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_RESPONSE):
            res = res and mp_not_chain_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_PRECEDENCE):
            res = res and mp_not_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_PRECEDENCE):
            res = res and mp_not_chain_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])

    return res


def run_all_mp_checkers_traces_model(trace, model_file):
    input = parse_decl(model_file)
    activities = input.activities
    res = True
    for constraint in input.checkers:
        if constraint['key'].startswith(EXISTENCE):
            res = res and mp_existence_checker(trace, True, constraint['attribute'],
                                                     constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ABSENCE):
            res = res and mp_absence_checker(trace, True, constraint['attribute'], constraint['condition'][0],
                                                   constraint['condition'][1])
        elif constraint['key'].startswith(INIT):
            res = res and mp_init_checker(trace, True, constraint['attribute'], constraint['condition'][0])
        elif constraint['key'].startswith(EXACTLY):
            res = res and mp_exactly_checker(trace, True, constraint['attribute'], constraint['condition'][0],
                                                   constraint['condition'][1])
        elif constraint['key'].startswith(CHOICE):
            res = res and mp_choice_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                  constraint['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(EXCLUSIVE_CHOICE):
            res = res and mp_exclusive_choice_checker(trace, True, constraint['attribute'].split(', ')[0]
            ['attribute'].split(', ')[1], constraint['condition'][0])
        elif constraint['key'].startswith(RESPONDED_EXISTENCE):
            res = res and mp_responded_existence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                               constraint['attribute'].split(', ')[1],
                                                               constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(RESPONSE):
            res = res and mp_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                    constraint['attribute'].split(', ')[1],
                                                    constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            res = res and mp_alternate_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_RESPONSE):
            res = res and mp_chain_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(PRECEDENCE):
            res = res and mp_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                      constraint['attribute'].split(', ')[1],
                                                      constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_PRECEDENCE):
            res = res and mp_alternate_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(CHAIN_PRECEDENCE):
            res = res and mp_chain_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                            constraint['attribute'].split(', ')[1],
                                                            constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(ALTERNATE_RESPONSE):
            res = res and mp_not_responded_existence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                   constraint['attribute'].split(', ')[1],
                                                                   constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_RESPONSE):
            res = res and mp_not_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                        constraint['attribute'].split(', ')[1],
                                                        constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_RESPONSE):
            res = res and mp_not_chain_response_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                              constraint['attribute'].split(', ')[1],
                                                              constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_PRECEDENCE):
            res = res and mp_not_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                          constraint['attribute'].split(', ')[1],
                                                          constraint['condition'][0], constraint['condition'][1])
        elif constraint['key'].startswith(NOT_CHAIN_PRECEDENCE):
            res = res and mp_not_chain_precedence_checker(trace, True, constraint['attribute'].split(', ')[0],
                                                                constraint['attribute'].split(', ')[1],
                                                                constraint['condition'][0], constraint['condition'][1])

    return res

