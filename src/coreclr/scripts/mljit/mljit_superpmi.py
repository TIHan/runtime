import subprocess
import os
import datetime
import re
import threading
import asyncio
import time
import multiprocessing
import random
import sys
import queue
import concurrent.futures
import json
import mljit_utils
import time
from types import SimpleNamespace
from dataclasses import dataclass
from threading import Thread
from queue import SimpleQueue
from subprocess import PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir
from os.path import isfile, join

# --------------------------------------------------------------------------------

core_root         = os.environ['CORE_ROOT']
log_path          = os.environ['DOTNET_MLJitLogPath']
superpmi_exe      = os.path.join(core_root, 'superpmi.exe') # TODO: Only works on windows, fix it for other OSes
clrjit_dll        = os.path.join(core_root, 'clrjit.dll') # TODO: Only works on windows, fix it for other OSes
mldump_txt        = os.path.join(log_path, "mldump.txt")
mldump_data_txt        = os.path.join(log_path, "mldump_data.txt")

# --------------------------------------------------------------------------------
# Utility
def absolute_path(path):
    return os.path.abspath(path)

def get_current_working_directory():
    return os.getcwd()

def get_files(dir):
    return [x for x in listdir(dir) if isfile(join(dir, x))]

def now():
    return datetime.datetime.now()

def run(args, working_directory = None, env=None):
    start_time = now()
    subprocess.run(args, shell = True, cwd = working_directory, env=env)
    return now() - start_time

def regex(pattern, text, groupNum=0):
    result = re.search(pattern, text)
    if result == None:
        return ""
    else:
        return result.group(groupNum).strip()

class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  # could alternatively raise an exception, depends on the use case
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result

# --------------------------------------------------------------------------------

@dataclass
class Method:
    perf_score: float
    num_cse_usages: int
    num_cse_candidates: int
    cse_seq: any
    spmi_index: int
    code_size: float
    log: any

# not used, but useful for getting a complete list of results from JITted functions.
def produce_mldump_file(corpus_file_path):
    run(f"{superpmi_exe} -jitoption JitStdOutFile={mldump_txt} -jitoption JitMetrics=1 {clrjit_dll} {corpus_file_path}")

def mldump_file_exists():
    return os.path.isfile(mldump_txt)

def parse_mldump_line(line):
    perf_score_pattern         = regex('(PerfScore|perf score) (\d+(\.\d+)?)', line, 2)
    num_cse_usages_pattern     = regex('num cse ([0-9]{1,})', line, 1)
    num_cse_candidates_pattern = regex('num cand ([0-9]{1,})', line, 1)
    cse_seq_pattern            = regex('seq ([0-9,]*)', line, 1)
    spmi_index_pattern         = regex('spmi index ([0-9]{1,})', line, 1)
    code_size_pattern          = regex('Total bytes of code ([0-9]{1,})', line, 1)

    is_valid = True

    cse_seq = []
    if cse_seq_pattern:
        cse_seq = list(map(lambda x: int(x), cse_seq_pattern.replace(' ', '').split(',')))

    perf_score = 10000000000.0
    if perf_score_pattern:
        try:
            perf_score = float(perf_score_pattern)
        except Exception:
            print(f'[mljit] ERROR: There was an error when parsing the \'perf_score\' from the JIT output: {perf_score_pattern}')
            is_valid = False
    else:
        print(f'[mljit] ERROR: \'(PerfScore|perf score)\' does not exist.')
        is_valid = False

    code_size = 10000000000.0
    if code_size_pattern:
        try:
            code_size = float(code_size_pattern)
        except Exception:
            print(f'[mljit] ERROR: There was an error when parsing the \'Total bytes of code\' from the JIT output: {code_size_pattern}')
            is_valid = False
    else:
        print(f'[mljit] ERROR: \'Total bytes of code\' does not exist.')
        is_valid = False

    num_cse_usages = 0
    if num_cse_usages_pattern:
        try:
            num_cse_usages = int(num_cse_usages_pattern)
        except Exception:
            print(f'[mljit] ERROR: There was an error when parsing the \'num cse\' from the JIT output: {num_cse_usages_pattern}')
            is_valid = False
    else:
        print(f'[mljit] ERROR: \'num cse\' does not exist.')
        is_valid = False

    num_cse_candidates = 0
    if num_cse_candidates_pattern:
        try:
            num_cse_candidates = int(num_cse_candidates_pattern)
        except Exception:
            print(f'[mljit] ERROR: There was an error when parsing the \'num cand\' from the JIT output: {num_cse_candidates_pattern}')
            is_valid = False
    else:
        print(f'[mljit] ERROR: \'num cand\' does not exist.')
        is_valid = False

    spmi_index = -1
    if num_cse_usages_pattern:
        try:
            spmi_index = int(spmi_index_pattern)
        except Exception:
            print(f'[mljit] ERROR: There was an error when parsing the \'spmi index\' from the JIT output: {spmi_index_pattern}')
            is_valid = False
    else:
        print(f'[mljit] ERROR: \'spmi index\' does not exist.')
        is_valid = False

    if is_valid:
        return Method(
                perf_score,
                num_cse_usages,
                num_cse_candidates,
                cse_seq,
                spmi_index,
                code_size,
                []
            )
    else:
        return None

def parse_mldump_filter(predicate, lines):
    def filter_(x):
        if x is None:
            return False
        else:
            return predicate(x)
    return filter(filter_, map(parse_mldump_line, lines))

def parse_mldump_file_filter_aux(path, predicate):
    dump_file = open(path, "r")
    lines = list(filter(lambda x: x.startswith('; Total bytes of code'), dump_file.readlines()))
    dump_file.close()
    return list(parse_mldump_filter(predicate, lines))

def parse_mldump_file_filter(predicate):
    return parse_mldump_file_filter_aux(mldump_txt, predicate)

def parse_mldump_data_file_filter(predicate):
    return parse_mldump_file_filter_aux(mldump_data_txt, predicate)

def parse_log_file(spmi_index, path):
    try:
        f = open(path)
        data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        f.close()
        return data
    except Exception as error:
        print(f'[mljit] ERROR: There was an error when parsing the training log file \"{path}\" from the JIT output on spmi_index {spmi_index}:\n{error}')
        return []

# --------------------------------------------------------------------------------

# This is used to ignore spmi indices that cause a time-out. The time-out could be due to an error within SuperPMI.
# FIXME: This is global and doesn't differentiate between different corpus files.
ignore_indices = set()

def cleanup_logs():
    mljit_log_files = list(filter(lambda x: '_mljit_log_' in x, get_files(log_path)))
    for mljit_log_file in mljit_log_files:
        os.remove(os.path.join(log_path, mljit_log_file))

    try:
        os.remove(mldump_data_txt)
    except Exception:
        ()

# --------------------------------------------------------------------------------
def collect_data(corpus_file_path, spmi_methods=None, train_kind=0, verbose_log=False):
    if spmi_methods is not None:
        spmi_methods = list(filter(lambda x: not (x in ignore_indices), spmi_methods))

    start_time = time.time()

    # Default env vars
    superpmi_env = dict()

    # The two env vars below prevent warnings and other various verbose log messages
    # Any errors will still be printed though.
    superpmi_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    superpmi_env['TF_CPP_MIN_LOG_LEVEL'] = "3"

    superpmi_env['TMP'] = log_path
    superpmi_env['DOTNET_MLJitEnabled'] = "1"
    superpmi_env['DOTNET_MLJitSavedPolicyPath'] = os.environ['DOTNET_MLJitSavedPolicyPath']
    superpmi_env['DOTNET_MLJitSavedCollectPolicyPath'] = os.environ['DOTNET_MLJitSavedCollectPolicyPath']
    superpmi_env['DOTNET_MLJitTrainLogFile'] = os.path.join(log_path, f'_mljit_log_')
    superpmi_env['DOTNET_MLJitTrain'] = f'{train_kind}'

    cleanup_logs()

    # Parallelism is only available when a compile list is not specified.
    parallel = spmi_methods is None

    verbose_arg = '-v q'
    if verbose_log:
        verbose_arg = ''

    parallel_arg = ''
    if parallel:
        parallel_arg = '-p'

    compile_arg = ''
    if spmi_methods is not None:
        mcl_path = os.path.join(log_path, "mljit.mcl")
        mcl = "\n".join(map(lambda x: str(x), spmi_methods))
        compile_arg = f'-c {mcl_path}'
        f = open(mcl_path, "w")
        f.write(mcl)
        f.close()

    jit_std_out_file_arg = f'-jitoption JitStdOutFile={mldump_data_txt}'

    run(f"{superpmi_exe} {parallel_arg} {compile_arg} {verbose_arg} {jit_std_out_file_arg} -jitoption JitMetrics=1 {clrjit_dll} {corpus_file_path}", env=superpmi_env)

    methods = parse_mldump_data_file_filter(lambda _: True)

    results = []
    for method in methods:
        train_log_file_path = os.path.join(log_path, f'_mljit_log_{method.spmi_index}.json')
        if os.path.isfile(train_log_file_path):
            method.log = parse_log_file(method.spmi_index, train_log_file_path)

            # Log could be empty, so do not include this in the results.
            # As an example: this can mean that all the CSE candidates were not viable; the policy should not look at non-viable candidates.
            if method.log:
                results.append(method)
            else:
                print(f'[mljit] WARNING: spmi_index \'{method.spmi_index}\' log was empty. Ignoring it for future collections.')
                ignore_indices.add(method.spmi_index)
        else:
            print(f'[mljit] WARNING: spmi_index \'{method.spmi_index}\' did not have a log. Ignoring it for future collections.')
            ignore_indices.add(method.spmi_index)

    if spmi_methods is not None:
        for spmi_method in spmi_methods:
            was_found = False
            for method in methods:
                if method.spmi_index == spmi_method:
                    was_found = True
            if not was_found:
                print(f'[mljit] WARNING: spmi_index \'{spmi_method}\' was not found in the results, likely due to a SuperPMI/JIT error. Ignoring it for future collections.')
                ignore_indices.add(spmi_method)

    cleanup_logs()

    if not verbose_log:
        print(f"[mljit] SuperPMI finished in {(time.time() - start_time)} seconds.")

    return results
