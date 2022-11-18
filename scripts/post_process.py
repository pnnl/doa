
from doa.utils import find_olpercent
import argparse
from doa.args import update_args

def run(args):

   find_olpercent(data_path = args['data_path'], 
                    res_path = args['res_path'], 
                    target_name = args['target_name'], save_res=True, 
                    rerun_threshold_detection=True,
                    n_parts = args['n_parts'])

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='List the content of a folder')
   parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='../scripts/configs/main.yaml')
   args = parser.parse_args()
   args = update_args(args)


   run(args)