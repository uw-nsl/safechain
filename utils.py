import argparse
from ast import arg
import pytz
from datetime import datetime
import json
import datasets
from config import *
get_time = lambda: datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime('%y%m%d-%H%M')

chat_template_base = '''# Instruction: {instruction}

# Response: '''

# dataset load and parsing
def parse_strongreject(small=False):
    if small:
        from strong_reject.load_datasets import load_strongreject_small as load_data
    else:
        from strong_reject.load_datasets import load_strongreject as load_data
    data = load_data()

    harmful_inst = data['forbidden_prompt']
    harmful_target = [None for _ in harmful_inst]
    return harmful_inst, harmful_target

def parse_wildjailbreak(small=False):
    if small:
        split = '50'
    else:
        split = '250'

    harmful_inst = datasets.load_dataset('UWNSL/WildJailbreakEval', split)['train']['instruction']
    harmful_target =  [None for _ in harmful_inst]
    return harmful_inst, harmful_target


def get_args():
    '''for resp_gen and pipeline'''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=RSM_LIST[0], choices=RSM_LIST)
    argparser.add_argument('--data', type=str, default='strongreject', choices=EVAL_DATA)
    argparser.add_argument('--prompt', type=str, default='normal', help='setup for generation input')
    argparser.add_argument('--system', type=bool, default=DEFAULT_GEN_CONFIG['system'])
    argparser.add_argument('--temperature', type=float, default=DEFAULT_GEN_CONFIG['temperature'])
    argparser.add_argument('--topp', type=float, default=DEFAULT_GEN_CONFIG['topp'])
    argparser.add_argument('--topk', type=int, default=DEFAULT_GEN_CONFIG['topk'])
    argparser.add_argument('--max_tokens', type=int, default=DEFAULT_GEN_CONFIG['max_tokens'])
    argparser.add_argument('--repeat_n', type=int, default=DEFAULT_GEN_CONFIG['repeat_n'], help='number of sample per prompt input')
    argparser.add_argument('--n', type=int, default=-1, help='number of samples to use, -1 to use all')
    argparser.add_argument('--start_idx', type=int, default=0, help='start index of the samples to use')
    argparser.add_argument('--port', type=int, default=8000, help='output file name')
    argparser.add_argument('--think_budget', type=int, default=10000)
    argparser.add_argument('--enforce_num', type=int, default=10)
    args = argparser.parse_args()

    return args