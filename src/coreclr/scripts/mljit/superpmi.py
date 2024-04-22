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
from dataclasses import dataclass
from threading import Thread
from queue import SimpleQueue
from subprocess import PIPE

# --------------------------------------------------------------------------------

core_root        = os.environ['CORE_ROOT']
corpus_file_path = os.environ['DOTNET_MLJIT_CORPUS_FILE']
superpmi_exe     = os.path.join(core_root, 'superpmi.exe') # TODO: Only works on windows, fix it for other OSes
clrjit_dll       = os.path.join(core_root, 'clrjit.dll') # TODO: Only works on windows, fix it for other OSes

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
    perfScore: float
    numCse: int
    numCand: int
    seq: str
    spmi: int
    param: str
    likelihood: str
    baseLikelihood: str
    feature: str
    codeSize: float

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
            if not line.startswith("[streaming]") and not line.isspace():
                q.put(line)

    q = SimpleQueue()
    t = Thread(target=consume_output, args=(p, q))
    t.start()

    return (p, t, q, (l, s))

def create_many_superpmi_processes(clrjit_dll_path, mch_path):
    return [create_superpmi_process(clrjit_dll_path, mch_path) for x in range(superpmi_process_count)]

# not used, but useful for getting a complete list of results from JITted functions.
def jit_mldump(mch):
    run(f"{superpmi_exe} -jitoption JitStdOutFile=mldump.txt -jitoption JitMetrics=1 {clrjit_dll} {mch}")

def parse_mldump_line(line):
    perfScorePattern        = regex('(PerfScore|perf score) (\d+(\.\d+)?)', line, 2)
    numCsePattern           = regex('num cse ([0-9]{1,})', line, 1)
    numCandPattern          = regex('num cand ([0-9]{1,})', line, 1)
    seqPattern              = regex('seq ([0-9,]*)', line, 1)
    spmiPattern             = regex('spmi index ([0-9]{1,})', line, 1)
    paramPattern            = regex('updatedparams ([0-9\.,-e]{1,})', line, 1)
    likelihoodPattern       = regex('likelihoods ([0-9\.,-e]{1,})', line, 1)
    baseLikelihoodPattern   = regex('baseLikelihoods ([0-9\.,-e]{1,})', line, 1)
    featurePattern          = regex('features,([0-9]*,CSE #[0-9][0-9],[0-9\.,-e]{1,})', line)
    codeSizePattern         = regex('Total bytes of code ([0-9]{1,})', line, 1)

    return Method(
            float(perfScorePattern),
            int(numCsePattern),
            int(numCandPattern),
            seqPattern,
            int(spmiPattern),
            paramPattern,
            likelihoodPattern,
            baseLikelihoodPattern,
            featurePattern,
            float(codeSizePattern)
        )

def parse_mldump(lines):
    def filter_cse_methods(m):
        if m.numCse > 0:
            return True
        else:
            return False
    return filter(filter_cse_methods, map(parse_mldump_line, lines))

def parse_mldump_file():
    dump_file = open("mldump.txt", "r")
    lines = dump_file.readlines()
    dump_file.close()
    return parse_mldump(lines)

# --------------------------------------------------------------------------------

def superpmi_jit(superpmi_process, spmi_index):
    (p, _, q, _) = superpmi_process
    p.stdin.write(f'{spmi_index}\n')
    try:
        line = q.get(timeout=60)
        return parse_mldump_line(line)
    except Exception as error:
        print(f'There was an error when parsing a line from the JIT output:\n{error}')
        return None

def superpmi_terminate(superpmi_process):
    (p, t, _, _) = superpmi_process
    p.terminate()
    t.join()

def superpmi_is_busy(superpmi_process):
    (_, _, _, (l, _)) = superpmi_process
    return not l.is_set()

def superpmi_get_next_available_process(superpmi_processes):
    results = [x for x in superpmi_processes if not superpmi_is_busy(x)]
    while not results:
        # This will not happen if the ThreadPoolExecutor has the same number of workers as the number of superpmi_processes.
        (_, _, _, (l, _)) = superpmi_processes[random.randrange(0, len(superpmi_processes) - 1)]
        l.wait()
        results = [x for x in superpmi_processes if not superpmi_is_busy(x)]
    return results[0]

# --------------------------------------------------------------------------------

# TODO: Add more inputs.
def jit(spmi_index, superpmi_processes):
    p = None

    result = None
    while p is None:
        p = superpmi_get_next_available_process(superpmi_processes)
        (_, _, _, (l, s)) = p

        if s.acquire():
            l.clear()
            try:
                result = superpmi_jit(p, spmi_index)
            finally:
                l.set()
                s.release()

    return result

# --------------------------------------------------------------------------------
with concurrent.futures.ThreadPoolExecutor(max_workers=superpmi_process_count) as executor:
    def create_jit_task(spmi_index):
        return executor.submit(jit, spmi_index, superpmi_processes)

    # 300000 methods - dummy data
    indices = [37 for _ in range(3)]

    superpmi_processes = create_many_superpmi_processes(clrjit_dll, corpus_file_path)

    print("")
    print(f'core_root:\t\t{core_root}')
    print(f'clrjit_dll:\t\t{clrjit_dll}')
    print(f'superpmi_exe:\t\t{superpmi_exe}')
    print(f'Corpus:\t\t\t{corpus_file_path}')
    print(f'SuperPMI Process Count:\t{len(superpmi_processes)}')

    print("\nSuperPMI starting...\n")

    time_stamp = now()

    tasks = list(map(lambda i: create_jit_task(i), indices))

    def eval(task):
        try:
            return task.result()
        except Exception:
            return None
    results = list(filter(lambda x: x is not None, map(eval, tasks)))
    # TODO: do something with the results

    if not results:
        print("Warning: No method results were returned. Check if SuperPMI is being invoked correctly.")
    # else:
    #     for x in results:
    #         print(x)

print(f'\nSuperPMI stopping...\n')

for p in superpmi_processes:
    superpmi_terminate(p)

print(f'Finished in: {now() - time_stamp}')
# --------------------------------------------------------------------------------
