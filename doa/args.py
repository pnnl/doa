import yaml

def update_args(args):
   opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
   opt.update(vars(args))
   args = opt
   return args