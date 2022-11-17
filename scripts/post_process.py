
from doa.utils import find_olpercent
import argparse

def run(args):

   find_olpercent(data_path = args.data_path, 
                    res_path = args.res_path, 
                    target_name = args.target_name, save_res=True, 
                    rerun_threshold_detection=True,
                    n_parts = args.n_parts)

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='List the content of a folder')
   # Add the arguments
   parser.add_argument('--data-path', type=str, help='the path to list', default="/people/pana982/solubility/data/full_dataset/sets/train.csv")
   parser.add_argument('--n-parts', type=int, help='number of calculatio parts', default=1)
   parser.add_argument('--target-name', type=str, help='target proprty', default='log_sol')
   parser.add_argument('--res-path', type=str, help='the path to list', default='results/runs')

   args = parser.parse_args()

   run(args)