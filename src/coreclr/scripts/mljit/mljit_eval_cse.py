import os
import math
import logging
import mljit_superpmi
import mljit_utils

corpus_file_path = os.environ['DOTNET_MLJitCorpusFile']

# ---------------------------------------

if not mljit_superpmi.mldump_file_exists(corpus_file_path):
    print('[mljit] Producing mldump.txt...')
    mljit_superpmi.produce_mldump_file(corpus_file_path)
    print('[mljit] Finished producing mldump.txt')

def filter_cse_methods(m):
    return m.num_cse_candidates > 0 and m.perf_score > 0

spmi_indices = list(map(lambda x: x.spmi_index, mljit_superpmi.parse_mldump_file_filter(corpus_file_path, filter_cse_methods)))

# ---------------------------------------

print('[mljit] Collecting base methods...')
base_methods = mljit_superpmi.collect_data(corpus_file_path, spmi_indices, train_kind=0)
print('[mljit] Collecting diff methods...')
diff_methods = mljit_superpmi.collect_data(corpus_file_path, spmi_indices, train_kind=1)

print('[mljit] Comparing methods...')

num_improvements = 0
num_regressions = 0
num_same = 0
improvement_score = 0
regression_score = 0
improvement_size = 0
regression_size = 0

sum_log_improvement_score = 0
sum_log_regression_score = 0

method_dict = dict()
for base_method in base_methods:
    method_dict[base_method.spmi_index] = base_method

for diff_method in diff_methods:
    base_method = method_dict[diff_method.spmi_index]

    base_score = max(base_method.perf_score, 1.0)
    diff_score = max(diff_method.perf_score, 1.0)

    log_relative_score = math.log(diff_score / base_score)

    if abs(diff_score - base_score) < 0.01:
        num_same += 1
    elif diff_score < base_score:
        num_improvements += 1
        sum_log_improvement_score += log_relative_score
    elif base_score < diff_score:
        num_regressions += 1
        sum_log_regression_score += log_relative_score

    improvement_size += max(0, base_method.code_size - diff_method.code_size)
    regression_size += max(0, diff_method.code_size - base_method.code_size)

geomean_improvement_score = 0.0
if num_improvements > 0:
    geomean_improvement_score = math.exp(sum_log_improvement_score / num_improvements) - 1

geomean_regression_score = 0.0
if num_regressions > 0:
    geomean_regression_score = math.exp(sum_log_regression_score / num_regressions) - 1

print(mljit_utils.get_file_name(corpus_file_path))
print("{:,d} CSE contexts".format(len(diff_methods)))
print("({:,d} PerfScore improvements, {:,d} PerfScore regressions, {:,d} same PerfScore)".format(num_improvements, num_regressions, num_same))

improvement_size = int(improvement_size)
regression_size = int(regression_size)
if improvement_size > 0 and regression_size > 0:
    print("-{:,d}/+{:,d} bytes".format(improvement_size, regression_size))
elif improvement_size > 0:
    print("  -{:,d} bytes".format(improvement_size))
elif regression_size > 0:
    print("  +{:,d} bytes".format(regression_size))

if num_improvements > 0 and num_regressions > 0:
    print("{:.2f}%/+{:.2f}% PerfScore".format(geomean_improvement_score * 100, geomean_regression_score * 100))
elif num_improvements > 0:
    print("{:.2f}% PerfScore".format(geomean_improvement_score * 100))
elif num_regressions > 0:
    print("+{:.2f}% PerfScore".format(geomean_regression_score * 100))