import pandas as pd
from doa.utils import run_ol
import argparse

def run(args):

   df = pd.read_csv(args.data_path)
   df.reset_index(drop=True, inplace=True)
   features = df.drop(['smiles', args.target_name], axis=1).columns.values

   run_ol(1, args.n_runs, df, features)

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='List the content of a folder')
   # Add the arguments
   parser.add_argument('--data-path', type=str, help='the path to list', default="/people/pana982/solubility/data/full_dataset/sets/train.csv")
   parser.add_argument('--n-runs', type=int, help='the path to list', default=250)
   parser.add_argument('--target-name', type=str, help='the path to list', default='log_sol')
   args = parser.parse_args()

   run(args)
