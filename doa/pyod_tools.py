import pandas as pd

def get_pyod_ol_counts(times, all_data, pyod_dir):
    
    olcounts = {i:0 for i in all_data.smiles.values}
    all_ols=[]
    for name in times.name.values:
    #     with open(f'str_data_ols/old_evals/archive/{name}_ol_probs.pkl', 'rb') as f:
    #         res = pickle.load(f)

    #     assert res['name'] == name

    #     df = all_data.copy()

    #     df.loc[:, 'pp_lin'] = res['pp_lin']
    #     df.loc[:, 'pp_uni'] = res['pp_uni']
    #     df.loc[:, 'conf_lin'] = res['conf_lin']
    #     df.loc[:, 'conf_uni'] = res['conf_uni']

        df = pd.read_csv(f'{pyod_dir}/{name}_ol_probs.csv')    
        # select high confidence predictions
        
        """
        we only use predictions that have been made with confidence >= 0.99.
        out of those, we select the ones got classified as outliers.
        """
        df  = df[df.conf_lin >= .99]
        # select outliers
        ols = df[df.pp_uni == 1]

        if ols.shape[0] > 0:
            
            for i in ols.smiles:
                olcounts[i] +=1 
            print(name, ols.shape[0])


            all_ols.append(ols.smiles.values.tolist())

    return olcounts, all_ols