import argparse
import pm4py

from commons import shared_variables as shared
from src.commons.log_utils import LogData

log_name = 'Production.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=log_name, help='input log')
    args = parser.parse_args()

    log_path = shared.log_folder / log_name
    log_data = LogData(log_path)

    net, im, fm = pm4py.discover_petri_net_heuristics(log=log_data.log,
                                                      activity_key=log_data.act_name_key,
                                                      timestamp_key=log_data.timestamp_key,
                                                      case_id_key=log_data.case_name_key)

    alignments = pm4py.conformance_diagnostics_alignments(log_data.log, net, im, fm,
                                                          activity_key=log_data.act_name_key,
                                                          timestamp_key=log_data.timestamp_key,
                                                          case_id_key=log_data.case_name_key)

    print(f'th_fitness,% compliant traces')
    for i in range(1, 101):
        th = i / 100
        num_compliant_traces = len([alignment for alignment in alignments if alignment['fitness'] >= th])
        log_size = log_data.log[log_data.case_name_key].nunique()
        print(f'{th},{num_compliant_traces / log_size * 100}')

    # pm4py.view_petri_net(net, im, fm)
    pm4py.save_vis_petri_net(net, im, fm, shared.pn_folder / (log_name+'.jpg'))
    pm4py.write_pnml(net, im, fm, shared.pn_folder / (log_name+'.pnml'))
