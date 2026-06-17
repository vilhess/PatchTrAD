# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import argparse, time, os
import itertools

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import run_PatchTrAD
from TSB_AD.HP_list import Uni_algo_HP_dict

from benchmark_exp.configs import DEFAULT_PATHS
from benchmark_exp.utils import set_seed, print_cuda_info
from benchmark_exp.data_loader import load_file, train_split

# seeding
seed = 2024
set_seed(seed)

print_cuda_info()

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='HP Tuning')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_PATHS['dataset_dir'])
    parser.add_argument('--file_list', type=str, default=DEFAULT_PATHS['tuning_file_list'])
    parser.add_argument('--save_dir', type=str, default=DEFAULT_PATHS['tuning_save_dir'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Det_HP = Uni_algo_HP_dict['PatchTrAD']

    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    col_w = None  # set on first successful get_metrics call
    rows = []     # accumulate one row per (file, HP) so the CSV keeps every result

    for filename in file_list:

        print('Processing:{} by PatchTrAD'.format(filename))

        file_path = os.path.join(args.dataset_dir, filename)
        data, label, slidingWindow = load_file(file_path)
        data_train = train_split(data, filename)

        for params in combinations:

            output = run_PatchTrAD(data_train, data, **params)

            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                metrics = list(evaluation_result.values())
                if col_w is None:
                    col_w = ['file', 'HP'] + list(evaluation_result.keys())
            except Exception:
                metrics = [0] * 9

            row = [filename, params] + metrics
            rows.append(row)

            ## Temp Save
            if col_w is not None:
                save_path = f'{args.save_dir}/PatchTrAD.csv'
                pd.DataFrame(
                    rows,
                    columns=col_w,
                ).to_csv(save_path, index=False)
