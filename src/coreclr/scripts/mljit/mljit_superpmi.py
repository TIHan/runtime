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
import concurrent.futures
import json
from types import SimpleNamespace
from dataclasses import dataclass
from threading import Thread
from queue import SimpleQueue
from subprocess import PIPE

# --------------------------------------------------------------------------------

core_root         = os.environ['CORE_ROOT']
log_path          = os.environ['DOTNET_MLJitLogPath']
superpmi_exe      = os.path.join(core_root, 'superpmi.exe') # TODO: Only works on windows, fix it for other OSes
clrjit_dll        = os.path.join(core_root, 'clrjit.dll') # TODO: Only works on windows, fix it for other OSes
mldump_txt        = os.path.join(log_path, "mldump.txt")

# --------------------------------------------------------------------------------
# Utility
def absolute_path(path):
    return os.path.abspath(path)

def get_current_working_directory():
    return os.getcwd()

def now():
    return datetime.datetime.now()

def run(args, working_directory = None):
    start_time = now()
    subprocess.run(args, shell = True, cwd = working_directory)
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

superpmi_process_count = int(os.cpu_count() / 2) # divide by 2 because of hyper-threading.

@dataclass
class Method:
    is_valid: bool
    perfScore: float
    numCse: int
    numCand: int
    seq: any
    spmi_index: int
    # param: str
    # likelihood: str
    # baseLikelihood: str
    # feature: str
    codeSize: float
    log: any

def create_superpmi_process(clrjit_dll_path, mch_path):
    l = threading.Event()
    l.set()
    s = threading.Semaphore()

    # Default env vars
    superpmi_env = dict()

    # The two env vars below prevent warnings and other various verbose log messages
    # Any errors will still be printed though.
    superpmi_env['TF_ENABLE_ONEDNN_OPTS'] = "0"
    superpmi_env['TF_CPP_MIN_LOG_LEVEL'] = "3"
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
    t.start()

    return (p, t, q, (l, s))

def create_many_superpmi_processes(clrjit_dll_path, mch_path):
    return [create_superpmi_process(clrjit_dll_path, mch_path) for x in range(superpmi_process_count)]

# not used, but useful for getting a complete list of results from JITted functions.
def produce_mldump_file(corpus_file_path):
    run(f"{superpmi_exe} -jitoption JitStdOutFile={mldump_txt} -jitoption JitMetrics=1 {clrjit_dll} {corpus_file_path}")

def mldump_file_exists():
    return os.path.exists(mldump_txt)

def parse_mldump_line(line):
    perfScorePattern        = regex('(PerfScore|perf score) (\d+(\.\d+)?)', line, 2)
    numCsePattern           = regex('num cse ([0-9]{1,})', line, 1)
    numCandPattern          = regex('num cand ([0-9]{1,})', line, 1)
    seqPattern              = regex('seq ([0-9,]*)', line, 1)
    spmiPattern             = regex('spmi index ([0-9]{1,})', line, 1)
    # paramPattern            = regex('updatedparams ([0-9\.,-e]{1,})', line, 1)
    # likelihoodPattern       = regex('likelihoods ([0-9\.,-e]{1,})', line, 1)
    # baseLikelihoodPattern   = regex('baseLikelihoods ([0-9\.,-e]{1,})', line, 1)
    # featurePattern          = regex('features,([0-9]*,CSE #[0-9][0-9],[0-9\.,-e]{1,})', line)
    codeSizePattern         = regex('Total bytes of code ([0-9]{1,})', line, 1)

    is_valid = True

    seq = []
    if seqPattern:
        seq = list(map(lambda x: int(x), seqPattern.replace(' ', '').split(',')))

    perfScore = 10000000000.0
    if perfScorePattern:
        try:
            perfScore = float(perfScorePattern)
        except Exception:
            #print(f'There was an error when parsing the \'PerfScore\' from the JIT output: {perfScorePattern}')
            is_valid = False
    else:
        #print(f'\'PerfScore\' does not exist.')
        is_valid = False

    codeSize = 10000000000.0
    if codeSizePattern:
        try:
            codeSize = float(codeSizePattern)
        except Exception:
            #print(f'There was an error when parsing the \'Total bytes of code\' from the JIT output: {codeSizePattern}')
            is_valid = False
    else:
        #print(f'\'Total bytes of code\' does not exist.')
        is_valid = False

    numCse = 0
    if numCsePattern:
        try:
            numCse = int(numCsePattern)
        except Exception:
            #print(f'There was an error when parsing the \'num cse\' from the JIT output: {numCsePattern}')
            is_valid = False
    else:
        #print(f'\'num cse\' does not exist.')
        is_valid = False

    numCand = 0
    if numCandPattern:
        try:
            numCand = int(numCandPattern)
        except Exception:
            #print(f'There was an error when parsing the \'num cand\' from the JIT output: {numCandPattern}')
            is_valid = False
    else:
        #print(f'\'num cand\' does not exist.')
        is_valid = False

    spmi_index = -1
    if numCsePattern:
        try:
            spmi_index = int(spmiPattern)
        except Exception:
            #print(f'There was an error when parsing the \'spmi index\' from the JIT output: {spmiPattern}')
            is_valid = False
    else:
        #print(f'\'spmi index\' does not exist.')
        is_valid = False

    return Method(
            is_valid,
            perfScore,
            numCse,
            numCand,
            seq,
            spmi_index,
            # paramPattern,
            # likelihoodPattern,
            # baseLikelihoodPattern,
            # featurePattern,
            codeSize,
            []
        )

def parse_mldump_filter(predicate, lines):
    return filter(predicate, map(parse_mldump_line, lines))

def parse_mldump_file_filter(predicate):
    dump_file = open(mldump_txt, "r")
    lines = dump_file.readlines()
    dump_file.close()
    return list(parse_mldump_filter(predicate, lines))

def parse_log_file(spmi_index, path):
    try:
        f = open(path)
        data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        f.close()
        return data
    except Exception as error:
        print(f'There was an error when parsing the training log file from the JIT output on spmi_index {spmi_index}:\n{error}')
        return []

# --------------------------------------------------------------------------------

def superpmi_jit(superpmi_process, spmi_index, train_kind, cse_replay_seqs):
    (p, _, q, _) = superpmi_process
    log_file = os.path.join(log_path, f'_mljit_log_{p.pid}.json')
    cse_replay_seqs_option = ''
    if cse_replay_seqs:
        cse_replay_seqs_option = f'!JitReplayCSE=\"{cse_replay_seqs}\"'.replace('[', '').replace(']', '')
    train_option = f'!MLJitTrain={train_kind}'
    p.stdin.write(f'{spmi_index} !MLJitTrainLogFile={log_file} {cse_replay_seqs_option} {train_option}\n')
    try:
        line = ''
        try:
            line = q.get(timeout=10) # 10 seconds
        except Exception:
            return -1

        meth = parse_mldump_line(line)
        if meth.is_valid:
            meth.log = parse_log_file(spmi_index, log_file)
            if meth.numCse > 0 and not meth.log:
                print(f'Expected log info of CSEs for spmi_index {spmi_index}')
        else:
            print(f'Could not parse the metrics line on spmi_index {spmi_index}. Line: {line}')
        return meth
    except Exception as error:
        print(f'There was an error when parsing the JIT output on spmi_index {spmi_index}:\n{error}')
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
        # This will not happen if the ThreadPoolExecutor has the same number of workers as the number of superpmi_processes.
        (_, _, _, (l, _)) = superpmi_processes[random.randrange(0, len(superpmi_processes) - 1)]
        l.wait()
        results = [i for i in range(len(superpmi_processes)) if not superpmi_is_busy(superpmi_processes[i])]
    return (results[0], superpmi_processes[results[0]])

# --------------------------------------------------------------------------------

# TODO: Add more inputs?
def jit(clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs, superpmi_processes):
    p = None

    result = None
    while p is None:
        (pi, p) = superpmi_get_next_available_process(superpmi_processes)
        (_, _, _, (l, s)) = p

        if s.acquire():
            l.clear()
            try:
                result = superpmi_jit(p, spmi_index, train_kind, cse_replay_seqs)
                if result == -1:
                    print(f'[mljit] spmi_index {spmi_index} timed out. Terminating SuperPMI process...')
                    superpmi_terminate(p)
                    print("[mljit] Creating new SuperPMI process...")
                    superpmi_processes[pi] = create_superpmi_process(clrjit_dll, corpus_file_path)

                    tmpp = create_superpmi_process(clrjit_dll, corpus_file_path)

                    print(f"[mljit] Running isolated SuperPMI process for spmi_index {spmi_index}")
                    result = superpmi_jit(tmpp, spmi_index, train_kind, cse_replay_seqs)
                    superpmi_terminate(tmpp)
                    if result == -1:
                        print(f'[mljit] spmi_index {spmi_index} timed out in isolated SuperPMI process.')
                        result = None
                    p = -1
            finally:
                l.set()
                s.release()

    return result

# --------------------------------------------------------------------------------
def collect_data(corpus_file_path, spmi_methods, train_kind, verbose_log=False):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=superpmi_process_count) as executor:
        def create_jit_task(clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs):
            return executor.submit(jit, clrjit_dll, corpus_file_path, spmi_index, train_kind, cse_replay_seqs, superpmi_processes)

        superpmi_processes = create_many_superpmi_processes(clrjit_dll, corpus_file_path)

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

        def eval(task):
            try:
                return task.result()
            except Exception:
                return None
        results = list(filter(lambda x: x is not None, map(eval, tasks)))

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
