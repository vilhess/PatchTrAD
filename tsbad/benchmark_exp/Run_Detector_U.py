import pandas as pd
import argparse, time, os

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.model_wrapper import run_PatchTrAD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

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
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_PATHS['dataset_dir'])
    parser.add_argument('--file_list', type=str, default=DEFAULT_PATHS['file_list'])
    parser.add_argument('--save_dir', type=str, default=DEFAULT_PATHS['save_dir'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok = True)

    file_list = pd.read_csv(args.file_list)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict["PatchTrAD"]

    col_w = None  # set on first successful get_metrics call (or from loaded CSV when resuming)
    rows = []     # accumulate one row per file so the CSV keeps every result

    for i, filename in enumerate(file_list):

        print('Processing:{} by PatchTrAD'.format(filename))

        file_path = os.path.join(args.dataset_dir, filename)
        data, label, slidingWindow = load_file(file_path)
        data_train = train_split(data, filename)

        start_time = time.time()

        output = run_PatchTrAD(data_train, data, **Optimal_Det_HP)
        run_time = time.time() - start_time

        try:
            evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
            print('evaluation_result: ', evaluation_result)
            metrics = list(evaluation_result.values())
            if col_w is None:
                col_w = ['file', 'Time'] + list(evaluation_result.keys())
        except Exception:
            metrics = [0] * 9

        row = [filename, run_time] + metrics
        rows.append(row)
        ## Temp Save
        if col_w is not None:
            save_path = f'{args.save_dir}/PatchTrAD.csv'
            pd.DataFrame(
                rows,
                columns=col_w,
            ).to_csv(save_path, index=False)


        
    