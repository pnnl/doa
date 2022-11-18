import pandas as pd
from doa.detect import run_ol
from doa.args import update_args
import argparse


def run(args):

   df = pd.read_csv(args['data_path'])
   df.reset_index(drop=True, inplace=True)
   features = df.drop(['smiles', args['target_name'] ], axis=1).columns.values

   run_ol(part=args['run_id'], n_runs=args['n_runs'], df=df, features=features, 
         target_name=args['target_name'], res_path=args['res_path']) 



if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='List the content of a folder')
   parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='../scripts/configs/main.yaml')
   parser.add_argument('--run-id', type=int, help='unique identifier for the run', default=1)
   args = parser.parse_args()   
   args = update_args(args)


   run(args)
