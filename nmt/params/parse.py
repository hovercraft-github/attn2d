"""
Read config files and pass dict of parameters
"""
import sys
import os
import os.path as osp
import argparse
import yaml
from .utils import update, parse_densenet_params, create_logger


def parse_eval_params():
    """
    Parse parametres from config file for evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="configuration file")
    # Command line arguments
    parser.add_argument("-v", "--verbose", type=int,
                        default=1, help="code verbosity")
    parser.add_argument("-g", "--gpu_id", type=str,
                        default='0', help="gpu id")
    parser.add_argument("-b", "--beam_size", type=int,
                        default=5, help="beam size for decoding")
    parser.add_argument("-o", "--offset", type=int,
                        default=0, help="starting index used to visualize a specific batch")
    parser.add_argument("--read_length", type=int,
                        default=0, help="max length for loading")
    parser.add_argument("--max_length_a",  type=float,
                        default=0, help="Decode up to a*source_lenght + b : (a)")
    parser.add_argument("--max_length_b", type=int,
                        default=80, help="Decode up to a*source_lenght + b : (b)")
    parser.add_argument("-n", "--batch_size", type=int,
                        default=1, help="batch size for decoding")
    parser.add_argument("-l", "--last", action="store_true",
                        help="evaluate with the last checkpoint instead of the best one")
    parser.add_argument("--norm", action="store_true", help="Normalize scores by length")
    parser.add_argument("-m", "--max_samples", type=int,
                        default=100, help="Decode up to max_samples sequences")
    parser.add_argument("--block_ngram_repeat", type=int,
                        default=0, help="GNMT parameter")
    parser.add_argument("--length_penalty", "-p", type=float,
                        default=0, help="length penalty for GNMT")
    parser.add_argument("--length_penalty_mode", type=str,
                        default="wu", help="length penalty mode, either wu or avg for GNMTscorer")

    parser.add_argument("--stepwise_penalty", action="store_true")
    parser.add_argument("-s", "--split", type=str,
                        default="test", help="Split to evaluate")

    args = parser.parse_args()
    default_config_path = "config/default.yaml"
    if args.config:
        config = yaml.load(open(args.config))
        default_config_path = config.get('default_config', default_config_path)

    default_config = yaml.load(open(default_config_path))
    default_config = update(default_config, config)
    parser.set_defaults(**default_config)
    args = parser.parse_args()
    # Mkdir the model save directory
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    args = vars(args)
    args = parse_densenet_params(args)
    # Make sure the dirs exist:
    if not osp.exists(args['modelname']):
        sys.exit('Missing direcetory %s' % args['modelname'])
    # Create the logger
    logger = create_logger(args['modelname'] + "_eval",
                           '%s/eval.log' % args['modelname'])
    return args


def parse_params():
    """
    Parse parametres from config file for training
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")
    parser.add_argument("-v", "--verbose", type=int,
                        default=1, help="code verbosity")
    parser.add_argument("-g", "--gpu_id", type=str,
                        default='0', help="gpu id")
    args = parser.parse_args()
    default_config_path = "config/default.yaml"
    if args.config:
        config = yaml.load(open(args.config))
        default_config_path = config.get('default_config', default_config_path)

    default_config = yaml.load(open(default_config_path))
    default_config = update(default_config, config)
    parser.set_defaults(**default_config)
    args = parser.parse_args()
    args.eventname = 'events/' + args.modelname
    args.modelname = 'save/' + args.modelname
    # Make sure the dirs exist:
    if not osp.exists(args.eventname):
        os.makedirs(args.eventname)
    if not osp.exists(args.modelname):
        os.makedirs(args.modelname)
    # Create the logger
    logger = create_logger(args.modelname, '%s/train.log' % args.modelname)
    args = vars(args)
    args = parse_densenet_params(args)
    return args

