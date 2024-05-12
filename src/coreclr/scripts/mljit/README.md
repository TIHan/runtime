# MLJIT (Machine Learning JIT)
This project is intended to replicate or improve upon Andy's work for [MLCSE](https://github.com/dotnet/jitutils/blob/main/src/jit-rl-cse/README.md) using a state-of-the-art ML framework such as TensorFlow.

While we have not successfully got training to work as intended yet, we do have data flowing between the JIT and the Python side.

Note: **This only works for Windows** at the moment given the usage of the TensorFlow C API only links Windows-specific TensorFlow libraries. This could be fixed by adding the necessary CMake changes to include Linux TensorFlow libraries.

## Build Instructions

This assumes you know how to build the runtime and JIT.

1. Install TensorFlow C API
    - From this [link](https://www.tensorflow.org/install/lang_c), download the Windows CPU only lib [here](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.15.0.zip).
    - You will also need to download the Linux CPU only lib [here](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz). (will explain why below)
    - Create a new folder called `tensorflow` in `src/native/external`.
    - Extract contents from TensorFlow Windows CPU only lib zip file into `src/native/external/tensorflow`. 
    - When you try to build the JIT, you will receive errors regarding missing header files. For some reason, TensorFlow forgot to include these header files in the Windows library. To fix this, look in the TensorFlow Linux CPU only lib zip file for the header files and put them in the appropriate places in `src/native/external/tensorflow`. This is a pain, but it will just be more contents from the `tls` folder and the `tf_buffer.h` file.
    - With these steps done, you should be able to build the JIT now. Make sure it's a **Checked** build.
2. Python
    - Install Python 3.11 or later.
    - From command-line: 
        - Run `pip3 install tf_agents`. This should install all the necessary dependencies for using the `tf_agents` library, including TensorFlow.
        - Run `pip3 install tensorboard`.

From these steps, you should be able to build the JIT and run the python script `src/coreclr/scripts/mljit_train_cse.py` without dependency issues. For instructions on how to run `mljit_train_cse.py`, see below.

## Environment Variables

- `CORE_ROOT`
    - Set this your `Core_Root` folder after you have built the JIT. 
    - Used only in the Python scripts.
    - Example: `CORE_ROOT=C:\work\runtime\artifacts\tests\coreclr\windows.x64.Checked\Tests\Core_Root`.
- `DOTNET_MLJitCorpusFile`
    - This is the `.mch` file that will be used for training. The `.mch` file is used for replaying SuperPMI collections and is basically a collection of method contexts. 
    - Used only in the Python scripts.
    - Example: `DOTNET_MLJitCorpusFile=C:\work\mljit\libraries.pmi.windows.x64.checked.mch`
- `DOTNET_MLJitSavedPolicyPath`
    - This is the directory path that will store/save the TensorFlow's model. 
    - Used in both the JIT lib and Python scripts.
    - Example: `DOTNET_MLJitSavedPolicyPath=C:\work\mljit\saved_policy`
- `DOTNET_MLJitSavedCollectPolicyPath`
    - This is the directory path that will store/save the TensorFlow's model used for training.
    - Used in both the JIT lib and Python scripts.
    - Example: `DOTNET_MLJitSavedCollectPolicyPath=C:\work\mljit\saved_collect_policy`
- `DOTNET_MLJitLogPath`
    - This is the directory path that acts as a scratch pad to write/read `json` files that the JIT will produce when it records training logs which get used for ML training.
    - Used in both the JIT lib and Python scripts.
    - Example: `DOTNET_MLJitLogPath=C:\work\mljit`
- `DOTNET_MLJitEnabled`
    - Enables the execution of TensorFlow policies, including ones for training, in the JIT.
    - Used in the JIT lib, but set in `mljit_superpmi.py`.
    - Requires `DOTNET_MLJitLogPath` to be set with a valid path.
    - Example: `DOTNET_MLJitEnabled=1`
- `DOTNET_MLJitTrain`
    - When `DOTNET_MLJITEnabled=1`, these are the types of ways to execute policies.
    - Used in the JIT lib, but set in `mljit_superpmi.py`.
    - There are three modes:
        - `0` - Doesn't execute a policy, but records the inputs/outputs of the current CSE heuristic decisions. Requires `DOTNET_MLJitSavedCollectPolicyPath` to be set.
        - `1` - Executes the `collect_policy` to make CSE decisions. It's stochastic when it makes a decision which is why this mode is used for training, it's trying to be used for random exploration. Records the inputs/outputs when the policy gets executed. Requires `DOTNET_MLJitSavedCollectPolicyPath` to be set.
        - `2` - Executes the `policy` to make CSE decisions. This is meant to be the final policy produced after training and used for evaluation. Records the inputs/outputs when the policy gets executed. Requires `DOTNET_MLJitSavedPolicyPath` to be set.
    - Example: `DOTNET_MLJitTrain=1` for random exploration.
- `DOTNET_MLJitUseBC`
    - TODO

## `mljit_train_cse.py`

- Before executing `mljit_train_cse.py`, the following environment variables must be set:
```
CORE_ROOT=C:\work\runtime\artifacts\tests\coreclr\windows.x64.Checked\Tests\Core_Root
DOTNET_MLJitCorpusFile=C:\work\mljit\libraries.pmi.windows.x64.checked.mch
DOTNET_MLJitSavedPolicyPath=C:\work\mljit\saved_policy
DOTNET_MLJitSavedCollectPolicyPath=C:\work\mljit\saved_collect_policy
DOTNET_MLJitLogPath=C:\work\mljit
```

- Once the environment variables are set appropriately, you should simply be able to run the script: `python src\coreclr\scripts\mljit\mljit_train_cse.py`.
- The script will do the following:
    - On first run, will produce a `mldump.txt` by replaying the `.mch` file from `DOTNET_MLJitCorpusFile`.
    - From the `mldump.txt` file, will gather a list of all methods that contain CSEs.
    - Training will commence.
    - Once training is complete, it will then compare results from the baseline to see how many improvements and regressions were made from the final policy.

## TensorBoard

From [TensorFlow](https://www.tensorflow.org/tensorboard), it describes TensorBoard as:
```
TensorBoard provides the visualization and tooling needed for machine learning experimentation
```

Instructions:
- While training, open a new command prompt.
- Set environment variable `DOTNET_MLJitLogPath` to the same path as when you started training.
- Run `tensorboard --logdir %DOTNET_MLJitLogPath%`.
- When TensorBoard launches, it will print a URL, example: `http://localhost:6006/`. Go to that URL and you will see all the information about your training session.

## Useful Links

- https://github.com/google/ml-compiler-opt - MLJIT was inspired by MLGO
- [MLGO: a Machine Learning Guided Compiler Optimizations Framework](https://arxiv.org/abs/2101.04808) - Paper that goes into detail about MLGO and its inlining policy.
