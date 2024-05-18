import os
import mljit_superpmi

corpus_file_path = os.environ['DOTNET_MLJitCorpusFile']

baseline_methods = mljit_superpmi.collect_all_data(corpus_file_path, train_kind=0, parallel=False, verbose_log=True)
policy_methods = mljit_superpmi.collect_all_data(corpus_file_path, train_kind=1, parallel=False, verbose_log=True)

num_improvements = 0
num_regressions = 0
improvement_score = 0
regression_score = 0

for policy_method in policy_methods:
    for base_method in baseline_methods:
        if base_method.spmi_index == policy_method.spmi_index:
            if policy_method.perf_score < base_method.perf_score:
                num_improvements = num_improvements + 1
                improvement_score = improvement_score + (base_method.perf_score - policy_method.perf_score)
            elif policy_method.perf_score > base_method.perf_score:
                num_regressions = num_regressions + 1
                regression_score = regression_score + (policy_method.perf_score - base_method.perf_score)

print(f'[mljit] Total CSE methods: {len(policy_methods)}')
print(f'[mljit] Improvements:      {num_improvements}')
print(f'[mljit] Improvement Score: {improvement_score}')
print(f'[mljit] Regressions:       {num_regressions}')
print(f'[mljit] Regression Score:  {regression_score}')