from parsers import *
from mp_checkers.trace.mp_choice import *
from mp_checkers.trace.mp_existence import *
from mp_checkers.trace.mp_negative_relation import *
from mp_checkers.trace.mp_relation import *

from pm4py.objects.log.importer.xes import factory as xes_import_factory


def run_all_mp_checkers(log_name, trace_id, model_file, prefix):
    log = xes_import_factory.apply(XES_LOG_FOLDER + log_name + ".xes")
    input = parse_decl(model_file)
    activities = input.activities
    res = True
    for t in log:
        if t.attributes['concept:name'] == trace_id:
            trace = t
    for key, conditions in input.checkers.items():
        if key.startswith(EXISTENCE):
            res = res and mp_existence_checker(trace[:prefix], True, activities[0], conditions[0], conditions[1])
        elif key.startswith(ABSENCE):
            res = res and mp_absence_checker(trace[:prefix], True, activities[0], conditions[0], conditions[1])
        elif key.startswith(INIT):
            res = res and mp_init_checker(trace[:prefix], True, activities[0], conditions[0])
        elif key.startswith(EXACTLY):
            res = res and mp_exactly_checker(trace[:prefix], True, activities[0], conditions[0], conditions[1])
        elif key.startswith(CHOICE):
            res = res and mp_choice_checker(trace[:prefix], True, activities[0], activities[1], conditions[0])
        elif key.startswith(EXCLUSIVE_CHOICE):
            res = res and mp_exclusive_choice_checker(trace[:prefix], True, activities[0], activities[1], conditions[0])
        elif key.startswith(RESPONDED_EXISTENCE):
            res = res and mp_responded_existence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(RESPONSE):
            res = res and mp_response_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(ALTERNATE_RESPONSE):
            res = res and mp_alternate_response_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(CHAIN_RESPONSE):
            res = res and mp_chain_response_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(PRECEDENCE):
            res = res and mp_precedence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(ALTERNATE_PRECEDENCE):
            res = res and mp_alternate_precedence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(CHAIN_PRECEDENCE):
            res = res and mp_chain_precedence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(ALTERNATE_RESPONSE):
            res = res and mp_not_responded_existence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(NOT_RESPONSE):
            res = res and mp_not_response_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(NOT_CHAIN_RESPONSE):
            res = res and mp_not_chain_response_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(NOT_PRECEDENCE):
            res = res and mp_not_precedence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])
        elif key.startswith(NOT_CHAIN_PRECEDENCE):
            res = res and mp_not_chain_precedence_checker(trace[:prefix], True, activities[0], activities[1], conditions[0], conditions[1])

    return res
