import argparse
import pm4py

from commons import shared_variables as shared

log_name = 'Synthetic log labelled'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=log_name+'.xes', help='input log')
    args = parser.parse_args()

    log_path = shared.log_folder / args.log

    xes_log = pm4py.read_xes(str(log_path))
    net, im, fm = pm4py.discover_petri_net_heuristics(xes_log)

    alignments = pm4py.conformance_diagnostics_alignments(xes_log, net, im, fm)

    print(f'th_fitness,heuristics % compliant traces')
    for i in range(1, 101):
        th = i / 100
        num_compliant_traces = len([alignment for alignment in alignments if alignment['fitness'] >= th])
        log_size = xes_log['case:concept:name'].nunique()
        print(f'{th},{num_compliant_traces / log_size * 100}')

    # pm4py.view_petri_net(net, im, fm)
    pm4py.save_vis_petri_net(net, im, fm, shared.pn_folder / (log_name+'_heuristics.jpg'))
    pm4py.write_pnml(net, im, fm, shared.pn_folder / (log_name+'_heuristics.pnml'))
