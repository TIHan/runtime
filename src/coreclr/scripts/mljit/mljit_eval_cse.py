import os
import mljit_superpmi

corpus_file_path = os.environ['DOTNET_MLJitCorpusFile']

# ---------------------------------------

if not mljit_superpmi.mldump_file_exists():
    print('[mljit] Producing mldump.txt...')
    mljit_superpmi.produce_mldump_file(corpus_file_path)
    print('[mljit] Finished producing mldump.txt')

def filter_cse_methods(m):
    return m.num_cse_candidates > 0 and m.perf_score > 0

spmi_indices = list(map(lambda x: x.spmi_index, mljit_superpmi.parse_mldump_file_filter(filter_cse_methods)))

# ---------------------------------------

print('[mljit] Collecting baseline methods...')
baseline_methods = mljit_superpmi.collect_data(corpus_file_path, spmi_indices, train_kind=0)
print('[mljit] Collecting policy methods...')
policy_methods = mljit_superpmi.collect_data(corpus_file_path, spmi_indices, train_kind=1)

print('[mljit] Comparing baseline and policy methods...')

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