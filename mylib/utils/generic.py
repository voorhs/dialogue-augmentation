import yaml, os
from argparse import Namespace
from datetime import datetime


def dump_cli_args(args: Namespace, path_out=None):
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    now = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    path_out = os.path.join(path_out, f'cli_args_{now}.yml')
    
    yaml.dump(vars(args), open(path_out, 'w'))
