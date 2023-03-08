import argparse
import pm4py

from commons import shared_variables as shared

log_name = 'sepsis_cases_1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=log_name+'.xes', help='input log')
    args = parser.parse_args()

    log_path = shared.log_folder / args.log

    log_settings = shared.log_settings[log_name]
    activity_key = log_settings['act_name_key']
    timestamp_key = log_settings['time_key']
    case_id_key = 'case:'+log_settings['trace_name_key']

    xes_log = pm4py.read_xes(str(log_path))
    net, im, fm = pm4py.discover_petri_net_heuristics(log=xes_log,
                                                      activity_key=activity_key,
                                                      timestamp_key=timestamp_key,
                                                      case_id_key=case_id_key)

    alignments = pm4py.conformance_diagnostics_alignments(xes_log, net, im, fm,
                                                          activity_key=activity_key,
                                                          timestamp_key=timestamp_key,
                                                          case_id_key=case_id_key)

    print(f'th_fitness,% compliant traces')
    for i in range(1, 101):
        th = i / 100
        num_compliant_traces = len([alignment for alignment in alignments if alignment['fitness'] >= th])
        log_size = xes_log[case_id_key].nunique()
        print(f'{th},{num_compliant_traces / log_size * 100}')

    # pm4py.view_petri_net(net, im, fm)
    pm4py.save_vis_petri_net(net, im, fm, shared.pn_folder / (log_name+'.jpg'))
    pm4py.write_pnml(net, im, fm, shared.pn_folder / (log_name+'.pnml'))
