#!/usr/bin/env python3
#
# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
#
#
# Title: superpmi_fsharp.py
#
# Notes:
#
# Script to perform the superpmi collection while build the fsharp proto
# in https://github.com/dotnet/fsharp.

import argparse
import re
import sys
import stat
import os
import time

from shutil import copyfile
from coreclr_arguments import *
from jitutil import run_command, ChangeDir, TempDir

# Start of parser object creation.
is_windows = platform.system() == "Windows"
parser = argparse.ArgumentParser(description="description")

parser.add_argument("-dotnet_directory", help="Path to dotnet directory")
parser.add_argument("-input_directory", help="Path to input directory")
parser.add_argument("-superpmi_directory", help="Path to superpmi directory")
parser.add_argument("-core_root", help="Path to Core_Root directory")
parser.add_argument("-output_mch_path", help="Absolute path to the mch file to produce")


def setup_args(args):
    """ Setup the args for SuperPMI to use.

    Args:
        args (ArgParse): args parsed by arg parser

    Returns:
        args (CoreclrArguments)

    """
    coreclr_args = CoreclrArguments(args, require_built_core_root=False, require_built_product_dir=False,
                                    require_built_test_dir=False, default_build_type="Checked")

    coreclr_args.verify(args,
                        "dotnet_directory",
                        lambda dotnet_directory: os.path.isdir(dotnet_directory),
                        "dotnet_directory doesn't exist")

    coreclr_args.verify(args,
                        "input_directory",
                        lambda input_directory: os.path.isdir(input_directory),
                        "input_directory doesn't exist")

    coreclr_args.verify(args,
                        "superpmi_directory",
                        lambda superpmi_directory: os.path.isdir(superpmi_directory),
                        "superpmi_directory doesn't exist")

    coreclr_args.verify(args,
                        "output_mch_path",
                        lambda output_mch_path: not os.path.isfile(output_mch_path),
                        "output_mch_path already exist")

    coreclr_args.verify(args,
                        "core_root",
                        lambda core_root: os.path.isdir(core_root),
                        "core_root doesn't exist")

    return coreclr_args


def make_executable(file_name):
    """Make file executable by changing the permission

    Args:
        file_name (string): file to execute
    """
    if is_windows:
        return

    print("Inside make_executable")
    run_command(["ls", "-l", file_name])
    os.chmod(file_name,
             # read+execute for owner
             (stat.S_IRUSR | stat.S_IXUSR) |
             # read+execute for group
             (stat.S_IRGRP | stat.S_IXGRP) |
             # read+execute for other
             (stat.S_IROTH | stat.S_IXOTH))
    run_command(["ls", "-l", file_name])


def build_and_run(coreclr_args, output_mch_name):
    """Build the fsharp proto and run them under "superpmi collect"

    Args:
        coreclr_args (CoreClrArguments): Arguments use to drive
        output_mch_name (string): Name of output mch file name
    """
    arch = coreclr_args.arch
    python_path = sys.executable
    core_root = coreclr_args.core_root
    superpmi_directory = coreclr_args.superpmi_directory
    fsharp_directory = os.path.join(coreclr_args.input_directory, "fsharp")
    sdk_directory = os.path.join(coreclr_args.input_directory, "sdk")
    dotnet_directory = coreclr_args.dotnet_directory
    dotnet_exe = os.path.join(dotnet_directory, "dotnet.cmd")

    artifacts_directory = os.path.join(fsharp_directory, "artifacts")
    artifacts_packages_directory = os.path.join(artifacts_directory, "packages")
    project_file = os.path.join(fsharp_directory, "src", "FSharp.Core", "FSharp.Core.fsproj")

    # Workaround https://github.com/dotnet/sdk/issues/23430
    project_file = os.path.realpath(project_file)

    if is_windows:
        script_name = "run_fsharp.bat"
    else:
        print("Windows only supported")

    make_executable(dotnet_exe)

    # Start with a "dotnet --info" to see what we've got.
    run_command([dotnet_exe, "--info"])

    env_copy = os.environ.copy()

    # Disable ReadyToRun so we always JIT R2R methods and collect them
    collection_command = f"{dotnet_exe} build {project_file} -c Proto"

    # Generate the execution script in Temp location
    with TempDir() as temp_location:
        script_name = os.path.join(temp_location, script_name)

        contents = []
        # Unset the JitName so dotnet process will not fail
        if is_windows:
            contents.append("set COMPlus_JitName=superpmi-shim-collector.dll")
            contents.append(f"set SuperPMIShimLogPath={coreclr_args.output_mch_path}")
            contents.append(f"set SuperPMIShimPath={core_root}\clrjit.dll")
        else:
            print("Windows only supported")
        contents.append(collection_command)

        with open(script_name, "w") as collection_script:
            collection_script.write(os.linesep.join(contents))

        print()
        print(f"{script_name} contents:")
        print("******************************************")
        print(os.linesep.join(contents))
        print("******************************************")

        make_executable(script_name)

        run_command([os.path.join(sdk_directory, "eng\\dogfood.cmd"), "-configuration", "Release"])
       # run_command(script_name)

def main(main_args):
    """ Main entry point

    Args:
        main_args ([type]): Arguments to the script
    """
    coreclr_args = setup_args(main_args)

    all_output_mch_name = os.path.join(coreclr_args.output_mch_path + "_all.mch")
    build_and_run(coreclr_args, all_output_mch_name)
    if os.path.isfile(all_output_mch_name):
        pass
    else:
        print("No mch file generated.")

if __name__ == "__main__":
    args = parser.parse_args()
    sys.exit(main(args))
