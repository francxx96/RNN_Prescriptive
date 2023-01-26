import shared_variables
from shared_variables import get_int_from_unicode
from pm4py.objects.log.obj import Trace, Event, EventLog
import pm4py


def get_pn_fitness(pn_file: str, trace_id: str, activities, groups=None):
    log_xes = EventLog()
    trace_xes = Trace()
    trace_xes.attributes["concept:name"] = trace_id
    for i in range(len(activities)):
        event = Event()
        event["concept:name"] = shared_variables.act_encoding[get_int_from_unicode(activities[i])]
        if groups is not None:
            event["org:resource"] = shared_variables.res_encoding[get_int_from_unicode(groups[i])]
        trace_xes.append(event)

    log_xes.append(trace_xes)

    net, initial_marking, final_marking = pm4py.read_pnml(pn_file, auto_guess_final_marking=True)
    alignment = pm4py.conformance_diagnostics_alignments(log_xes, net, initial_marking, final_marking)[0]
    return alignment['fitness']

