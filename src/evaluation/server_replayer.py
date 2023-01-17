# import subprocess
# from py4j.java_gateway import JavaGateway, GatewayParameters

import shared_variables
from shared_variables import get_int_from_unicode
from mp_checkers.test_mp_checkers_traces import run_all_mp_checkers_traces, run_all_mp_checkers_traces_model
from pm4py.objects.log import obj as lg
import pm4py
from datetime import datetime, timedelta


# class ServerReplayer:
"""def __init__(self, port, python_port):
    self._port = port
    self._python_port = python_port

    self._gateway, self._traces_checker = None, None
    self._server_process = None

def start_server(self):
    server_start_instructions = ['java', '-jar']
    server_args = ['../LTLCheckForTraces.jar', str(self._port), str(self._python_port)]

    self._server_process = subprocess.Popen(server_start_instructions + list(server_args))

    self._gateway = JavaGateway(gateway_parameters=GatewayParameters(port=self._port),
                                python_proxy_port=self._python_port)
    self._traces_checker = self._gateway.entry_point

def stop_server(self):
    self._server_process.terminate()
    self._server_process.wait()

def verify_with_data(self, model_file, trace_id, activities, groups, times, prefix=0):
    activities_java = self._gateway.jvm.java.util.ArrayList()
    groups_java = self._gateway.jvm.java.util.ArrayList()
    times_java = self._gateway.jvm.java.util.ArrayList()

    for i in range(prefix, len(activities)):
        activities_java.append(str(get_int_from_unicode(activities[i])))
        groups_java.append(str(get_int_from_unicode(groups[i])))
        times_java.append(times[i])
    if not activities_java:
        return False

    return self._traces_checker.isTraceWithDataViolated(model_file, trace_id, activities_java, groups_java,
                                                        times_java)

def verify_with_elapsed_time(self, model_file, trace_id, activities, groups, elapsed_times, times, prefix=0):

    activities_java = self._gateway.jvm.java.util.ArrayList()
    groups_java = self._gateway.jvm.java.util.ArrayList()
    elapsed_times_java = self._gateway.jvm.java.util.ArrayList()
    times_java = self._gateway.jvm.java.util.ArrayList()

    for i in range(prefix, len(activities)):
        activities_java.append(str(get_int_from_unicode(activities[i])))
        groups_java.append(str(get_int_from_unicode(groups[i])))
        elapsed_times_java.append(str(get_int_from_unicode(elapsed_times[i])))
        times_java.append(times[i])
    if not activities_java:
        return False

    return self._traces_checker.isTraceWithElapsedTimeViolated(model_file, trace_id, activities_java,
                                                               groups_java, elapsed_times_java, times_java)

def test_analysis(self):
    self._traces_checker.testAnalysis()

def verify_formula_as_compliant(self, trace, formula, prefix=0):
    trace_new = self._gateway.jvm.java.util.ArrayList()
    for i in range(prefix, len(trace)):
        trace_new.append(str(get_int_from_unicode(trace[i])))
    if not trace_new:
        return False
    ver = self._traces_checker.isTraceViolated(formula, trace_new) is False
    return ver"""

'''
def verify_with_data(model_file, trace_id, activities, groups, times):
    model_file = model_file.replace(".xml", ".decl")
    trace_xes = lg.Trace()
    trace_xes.attributes["concept:name"] = trace_id
    for i in range(len(activities)):
        event = lg.Event()
        event["concept:name"] = str(get_int_from_unicode(activities[i]))
        event["time:timestamp"] = times[i]
        event["org:resource"] = get_int_from_unicode(groups[i])
        trace_xes.append(event)
    return run_all_mp_checkers_traces_model(trace_xes, model_file)
'''


def verify_with_data(pn_file: str, trace_id: str, activities, groups, times):
    log_xes = lg.EventLog()
    trace_xes = lg.Trace()
    trace_xes.attributes["concept:name"] = trace_id
    for i in range(len(activities)):
        event = lg.Event()
        event["concept:name"] = shared_variables.act_encoding[get_int_from_unicode(activities[i])]
        event["time:timestamp"] = times[i]
        event["org:resource"] = shared_variables.res_encoding[get_int_from_unicode(groups[i])]
        trace_xes.append(event)

    log_xes.append(trace_xes)

    net, initial_marking, final_marking = pm4py.read_pnml(pn_file, auto_guess_final_marking=True)
    alignment = pm4py.conformance_diagnostics_alignments(log_xes, net, initial_marking, final_marking)[0]
    return alignment['fitness'] >= 1


def verify_with_elapsed_time(model_file, trace_id, activities, groups, times, elapsed_times, prefix=0):
    model_file = model_file.replace(".xml", ".decl")
    trace_xes = lg.Trace()
    trace_xes.attributes["concept:name"] = trace_id
    for i in range(len(activities)):
        event = lg.Event()
        event["concept:name"] = str(get_int_from_unicode(activities[i]))
        event["time:timestamp"] = times[i]
        event["org:resource"] = get_int_from_unicode(groups[i])
        trace_xes.append(event)
    return run_all_mp_checkers_traces_model(trace_xes, model_file)


def verify_formula_as_compliant(idx, trace, log_name, prefix, groups):
    trace_xes = lg.Trace()
    trace_xes.attributes["concept:name"] = idx
    c = 0
    for i in range(len(trace)):
        #import pdb
        #pdb.set_trace()
        event = lg.Event()
        event["concept:name"] = str(get_int_from_unicode(trace[i]))
        event["time:timestamp"] = (datetime.now() + timedelta(hours=c)).timestamp()
        if groups is not None:
            event["org:resource"] = get_int_from_unicode(groups[i])
        trace_xes.append(event)
        c += 1
    return run_all_mp_checkers_traces(trace_xes, log_name)


def verify_formula_ivan(idx, trace, log_name, groups, bk_type):
    if bk_type == "declare":
        log_name = log_name.replace(".xml", ".decl")
    trace_xes = lg.Trace()
    trace_xes.attributes["concept:name"] = idx
    c = 0
    for i in range(len(trace)):
        event = lg.Event()
        event["concept:name"] = str(get_int_from_unicode(trace[i]))
        event["time:timestamp"] = (datetime.now() + timedelta(hours=c)).timestamp()
        if groups is not None:
            event["org:resource"] = get_int_from_unicode(groups[i])
        trace_xes.append(event)
        c += 1
    if bk_type != "declare":
        return run_all_mp_checkers_traces(trace_xes, log_name)
    else:
        return run_all_mp_checkers_traces_model(trace_xes, log_name)