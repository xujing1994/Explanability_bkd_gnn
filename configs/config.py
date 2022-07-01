import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # argument for graph classification
    parser.add_argument('--batch_size', type=int, default=32, help="local batch size: B") ##
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details") ##
    parser.add_argument('--target_label', type=int, default=0, help='target label') ##
    parser.add_argument('--poisoning_intensity', type=float, default=0.2, help='frac of training dataset to be injected trigger') ##
    parser.add_argument('--frac_of_avg', type=float, default=0.2, help='frac of avg nodes to be injected the trigger') ##
    parser.add_argument('--bkd_size', type=int, default=5, help='number of nodes for each trigger')
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger') ##
    parser.add_argument('--filename', type = str, default = "", help='output file') ##
    parser.add_argument('--seed', type=int, default=0, help='seed') ##
    parser.add_argument('--clean_label', action='store_true', help='whether apply clean label backdoor attack or not')

    # argument for node classification
    parser.add_argument('--train_type', type=str, default='attack')
    parser.add_argument('--node_gat_config', type=str, default='configs/node_gat_config.json')
    parser.add_argument('--node_gcn_config', type=str, default='configs/node_gcn_config.json')
    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--trig_feat_val', type=float, default=1.0)
    parser.add_argument('--trig_feat_wid', type=int, default=10)

    args = parser.parse_args()
    return args
