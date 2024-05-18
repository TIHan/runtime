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
from types import SimpleNamespace
from dataclasses import dataclass
from threading import Thread
from queue import SimpleQueue
from subprocess import PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed

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

superpmi_process_count = int(os.cpu_count() / 2) # divide by 2 because hyper-threading
if superpmi_process_count == 0:
    superpmi_process_count = 1

@dataclass
class Method:
    perf_score: float
    num_cse_usages: int
    num_cse_candidates: int
    cse_seq: any
    spmi_index: int
    code_size: float
    log: any

def create_superpmi_process(clrjit_dll_path, mch_path, mljit_enabled):
    l = threading.Event()
    l.set()
    s = threading.Semaphore()

    # Default env vars
    superpmi_env = dict()

    # The two env vars below prevent warnings and other various verbose log messages
    # Any errors will still be printed though.
    superpmi_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    superpmi_env['TF_CPP_MIN_LOG_LEVEL'] = "3"

    if mljit_enabled:
        superpmi_env['DOTNET_MLJitEnabled'] = "1"
        superpmi_env['DOTNET_MLJitSavedPolicyPath'] = os.environ['DOTNET_MLJitSavedPolicyPath']
        superpmi_env['DOTNET_MLJitSavedCollectPolicyPath'] = os.environ['DOTNET_MLJitSavedCollectPolicyPath']

    superpmi_args = [
            superpmi_exe, 
            '-v', 'q',
            '-jitoption', 'JitMetrics=1', 
            '-streaming', 'stdin',
            f'\"{mch_path}\"',
            f'\"{clrjit_dll_path}\"'
        ]
    
    superpmi_args_joined = " ".join(superpmi_args)

    p = subprocess.Popen(
        superpmi_args_joined,
        stdin=PIPE, stdout=PIPE, stderr=PIPE,
        text=True, 
        bufsize=1, 
        universal_newlines=True,
        env=superpmi_env)

    def consume_output(p, q):
        while p.poll() is None:
            line = p.stdout.readline()
            # Uncomment below for debugging
            # if line:
            #     print(line)
            if line.startswith("; Total bytes of code"):
                q.put(line)

    q = SimpleQueue()
    t = Thread(target=consume_output, args=(p, q))
    t.daemon = True
    t.start()

    return (p, t, q, (l, s))

def create_many_superpmi_processes(clrjit_dll_path, mch_path, mljit_enabled):
    return [create_superpmi_process(clrjit_dll_path, mch_path, mljit_enabled) for _ in range(superpmi_process_count)]

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

def superpmi_jit(superpmi_process, spmi_index, train_kind, cse_replay_seqs):
    (p, _, q, _) = superpmi_process
    log_file = os.path.join(log_path, f'_mljit_log_  {spmi_index}.json')
    log_file_to_pass = os.path.join(log_path, f'_mljit_log_')
    cse_replay_seqs_option = ''
    if cse_replay_seqs:
        cse_replay_seqs_option = f'!JitReplayCSE=\"{cse_replay_seqs}\"'.replace('[', '').replace(']', '')
    train_option = f'!MLJitTrain={train_kind}'
    p.stdin.write(f'{spmi_index} !MLJitTrainLogFile={log_file_to_pass} {cse_replay_seqs_option} {train_option}\n')
    try:
        line = ''
        try:
            line = q.get(timeout=10) # 10 seconds
        except queue.Empty:
            print('[mljit] WARNING: Empty queue for SuperPMI process')
            return None
        except KeyboardInterrupt:
            return None
        except Exception:
            return -1

        meth = parse_mldump_line(line)
        if meth.is_valid:
            if train_kind != None:
                if os.path.isfile(log_file):
                    meth.log = parse_log_file(spmi_index, log_file)
                    if meth.num_cse_usages > 0 and not meth.log:
                        print(f'[mljit] WARNING: Expected log info of CSEs for spmi_index {spmi_index}')
                else:
                    print(f'[mljit] WARNING: Training log file not found for spmi_index {spmi_index}')
        else:
            print(f'[mljit] ERROR: Could not parse the metrics line on spmi_index {spmi_index}. Line: {line}')
        return meth
    except Exception as error:
        print(f'[mljit] ERROR: There was an error when parsing the JIT output on spmi_index {spmi_index}:\n{error}')
        return None
    finally:
        try:
            os.remove(log_file)
        except Exception as error:
            ()
            #print(f'There was an error when removing the log file:\n{error}')


def superpmi_terminate(superpmi_process):
    (p, t, _, _) = superpmi_process
    p.terminate()
    t.join()

def superpmi_is_busy(superpmi_process):
    (_, _, _, (l, _)) = superpmi_process
    return not l.is_set()

def superpmi_get_next_available_process(superpmi_processes):
    results = [i for i in range(len(superpmi_processes)) if not superpmi_is_busy(superpmi_processes[i])]
    while not results:
        print('[mljit] WARNING: Not enough SuperPMI processes')
        # This will not happen if the ThreadPoolExecutor has the same number of workers as the number of superpmi_processes.
        (_, _, _, (l, _)) = superpmi_processes[random.randrange(0, len(superpmi_processes) - 1)]
        l.wait()
        results = [i for i in range(len(superpmi_processes)) if not superpmi_is_busy(superpmi_processes[i])]
    return (results[0], superpmi_processes[results[0]])

# --------------------------------------------------------------------------------

# This is used to ignore spmi indices that cause a time-out. The time-out could be due to an error within SuperPMI.
# FIXME: This is global and doesn't differentiate between different corpus files.
ignore_indices = set()

def jit(clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs, superpmi_processes):
    if spmi_index in ignore_indices:
        return None
    
    result = None

    p = None
    while p is None:
        (pi, p) = superpmi_get_next_available_process(superpmi_processes)
        (_, _, _, (l, s)) = p

        if s.acquire():
            l.clear()
            try:
                result = superpmi_jit(p, spmi_index, train_kind, cse_replay_seqs)
                if result == -1:
                    print(f'[mljit] WARNING: spmi_index {spmi_index} timed out. Terminating SuperPMI process...')
                    superpmi_terminate(p)
                    result = None
                    ignore_indices.add(spmi_index)
                    print("[mljit] Creating new SuperPMI process...")
                    superpmi_processes[pi] = create_superpmi_process(clrjit_dll, corpus_file_path, train_kind != None)
                    p = -1
                if result is not None:
                    if result.spmi_index != spmi_index:
                        print(f'[mljit] ERROR: spmi_index does not match')
            finally:
                l.set()
                s.release()

    return result

# --------------------------------------------------------------------------------
# 'train_kind' correpsonds to 'DOTNET_MLJitTrain'
def collect_data_old(corpus_file_path, spmi_methods, train_kind=None, verbose_log=False):
    spmi_methods = list(filter(lambda x: not (x in ignore_indices), spmi_methods))
    results = []
    with ThreadPoolExecutor(max_workers=superpmi_process_count) as executor:
        def create_jit_task(clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs):
            return executor.submit(jit, clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs, superpmi_processes)

        superpmi_processes = create_many_superpmi_processes(clrjit_dll, corpus_file_path, train_kind != None)

        if verbose_log:
            print("[mljit] Collecting data...")
            print(f'[mljit] core_root:\t\t{core_root}')
            print(f'[mljit] clrjit_dll:\t\t{clrjit_dll}')
            print(f'[mljit] superpmi_exe:\t\t{superpmi_exe}')
            print(f'[mljit] Corpus:\t\t\t{corpus_file_path}')
            print(f'[mljit] SuperPMI Process Count:\t{len(superpmi_processes)}')
            print(f'[mljit] SuperPMI starting... Method count: {len(spmi_methods)} \n')

        time_stamp = now()

        tasks = list(map(lambda x: create_jit_task(clrjit_dll, corpus_file_path, x, train_kind, []), spmi_methods))

        eval_length = len(spmi_methods)
        eval_count = 0
        results = []
        try:
            for task in as_completed(tasks):
                try:
                    result = task.result()
                    if result is not None:
                        results = results + [result]
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    print(f"[mljit] ERROR: Unable to get result due to:\n{error}")
                finally:
                    eval_count = eval_count + 1
                    mljit_utils.print_progress_bar(eval_count, eval_length)
        except KeyboardInterrupt:
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
            for p in superpmi_processes:
                superpmi_terminate(p)
            raise

        if not results:
            print("[mljit] WARNING: No method results were returned. Check if SuperPMI is being invoked correctly.")

    if verbose_log:
        print(f'\n[mljit] SuperPMI stopping...')

    for p in superpmi_processes:
        superpmi_terminate(p)

    if verbose_log:
        print(f'[mljit] SuperPMI finished in: {now() - time_stamp}')
        print(f'[mljit] SuperPMI result count: {len(results)}\n')
    return results

# --------------------------------------------------------------------------------
def collect_all_data(corpus_file_path, train_kind=0, parallel=False, verbose_log=False):
    # Default env vars
    superpmi_env = dict()

    # The two env vars below prevent warnings and other various verbose log messages
    # Any errors will still be printed though.
    superpmi_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    superpmi_env['TF_CPP_MIN_LOG_LEVEL'] = "3"

    superpmi_env['DOTNET_MLJitEnabled'] = "1"
    superpmi_env['DOTNET_MLJitSavedPolicyPath'] = os.environ['DOTNET_MLJitSavedPolicyPath']
    superpmi_env['DOTNET_MLJitSavedCollectPolicyPath'] = os.environ['DOTNET_MLJitSavedCollectPolicyPath']
    superpmi_env['DOTNET_MLJitTrainLogFile'] = os.path.join(log_path, f'_mljit_log_')
    superpmi_env['DOTNET_MLJitTrain'] = f'{train_kind}'

    try:
        os.remove(mldump_data_txt)
    except Exception:
        ()

    verbose_arg = '-v q'
    if verbose_log:
        verbose_arg = ''

    parallel_arg = ''
    if parallel:
        parallel_arg = '-p'

    run(f"{superpmi_exe} {parallel_arg} {verbose_arg} -jitoption JitStdOutFile={mldump_data_txt} -jitoption JitMetrics=1 {clrjit_dll} {corpus_file_path}", env=superpmi_env)

    methods = parse_mldump_data_file_filter(lambda x: True)

    try:
        os.remove(mldump_data_txt)
    except Exception:
        ()

    results = []
    for method in methods:
        train_log_file_path = os.path.join(log_path, f'_mljit_log_{method.spmi_index}.json')
        if os.path.isfile(train_log_file_path):
            method.log = parse_log_file(method.spmi_index, train_log_file_path)
            os.remove(train_log_file_path)
            if method.log:
                results.append(method)

    return results

# --------------------------------------------------------------------------------
def collect_data(corpus_file_path, spmi_methods, train_kind=0, verbose_log=False):
     # Default env vars
    superpmi_env = dict()

    spmi_methods = list(filter(lambda x: not (x in ignore_indices), spmi_methods))
    # The two env vars below prevent warnings and other various verbose log messages
    # Any errors will still be printed though.
    superpmi_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    superpmi_env['TF_CPP_MIN_LOG_LEVEL'] = "3"

    superpmi_env['DOTNET_MLJitEnabled'] = "1"
    superpmi_env['DOTNET_MLJitSavedPolicyPath'] = os.environ['DOTNET_MLJitSavedPolicyPath']
    superpmi_env['DOTNET_MLJitSavedCollectPolicyPath'] = os.environ['DOTNET_MLJitSavedCollectPolicyPath']
    superpmi_env['DOTNET_MLJitTrainLogFile'] = os.path.join(log_path, f'_mljit_log_')
    superpmi_env['DOTNET_MLJitTrain'] = f'{train_kind}'

    mcl_path = os.path.join(log_path, "mljit.mcl")
    mcl = "\n".join(map(lambda x: str(x), spmi_methods))

    try:
        os.remove(mldump_data_txt)
    except Exception:
        ()

    f = open(mcl_path, "w")
    f.write(mcl)
    f.close()

    compile_arg = f'-c {mcl_path}'

    verbose_arg = '-v q'
    if verbose_log:
        verbose_arg = ''

    run(f"{superpmi_exe} {verbose_arg} {compile_arg} -jitoption JitStdOutFile={mldump_data_txt} -jitoption JitMetrics=1 {clrjit_dll} {corpus_file_path}", env=superpmi_env)

    methods = parse_mldump_data_file_filter(lambda x: True)

    try:
        os.remove(mldump_data_txt)
    except Exception:
        ()

    results = []
    for method in methods:
        train_log_file_path = os.path.join(log_path, f'_mljit_log_{method.spmi_index}.json')
        if os.path.isfile(train_log_file_path):
            method.log = parse_log_file(method.spmi_index, train_log_file_path)
            os.remove(train_log_file_path)

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

    for spmi_method in spmi_methods:
        was_found = False
        for method in methods:
            if method.spmi_index == spmi_method:
                was_found = True
        if not was_found:
            print(f'[mljit] WARNING: spmi_index \'{spmi_method}\' was not found in the results, likely due to a SuperPMI or JIT error. Ignoring it for future collections.')
            ignore_indices.add(spmi_method)

    return results
# --------------------------------------------------------------------------------
