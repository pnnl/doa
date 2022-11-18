import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import pathlib
import os
from doa.detect import collect_prediction_results
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from mordred import Calculator, descriptors


def create_dataset(csv_path, smiles_column, logs_column, save_path):

    df = pd.read_csv(csv_path)
    # https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8


    calc = Calculator(descriptors, ignore_3D=True)

    mols = [Chem.MolFromSmiles(smiles) for smiles in  df[smiles_column]]
    df_des = calc.pandas(mols)
    sc = StandardScaler()
    x = sc.fit_transform(df_des)
    df_des.drop(df_des.columns[np.where(np.isnan(x))[1]], axis=1, inplace=True)

    df_des['smiles'] = df[smiles_column]
    df_des['log_sol'] = df[logs_column]
    df_des.to_csv(save_path, index=False)

    return df_des









def save_get_outliers(save=False, th_file_id=None, all_data=None):
    if save:
        
        df_ol = pd.read_csv(f"{config.DETECT_DIR}/ols_with_olpercent_RES/outliers_olness_p{th_file_id}.csv")
        df_nol = pd.read_csv(f"{config.DETECT_DIR}/ols_with_olpercent_RES/non_outliers_olness_p{th_file_id}.csv")

        # os.mkdir(f"{config.DETECT_DIR}/outliers")
        df_ol.to_csv(f"{config.DETECT_DIR}/outliers/outliers.csv", index=False)
        df_nol.to_csv(f"{config.DETECT_DIR}/outliers/non_outliers.csv", index=False)

    else:
        
        df_ol = pd.read_csv(f"{config.DETECT_DIR}/outliers/outliers.csv")
        df_nol = pd.read_csv(f"{config.DETECT_DIR}/outliers/non_outliers.csv")
        
    if all_data.shape[0] == (df_ol.shape[0] + df_nol.shape[0]):
        print(" the ol + nol molecules == all molecules")
        
    print("outlier percentage: ", 100*df_ol.shape[0]/(df_ol.shape[0] + df_nol.shape[0]) )
    return df_ol, df_nol
    

def plot_olness(enpls=None, save_loc="./images/unsup/mean_sdev_th.png"):
    
    plt.plot(enpls['means_sc'], enpls['sdev_sc'], 'o', color='#DD361F')
    a = enpls[(enpls['means_sc'] < .6) & (enpls['sdev_sc'] <.6)]
    plt.plot(a['means_sc'], a['sdev_sc'], 'o', color='#1C6EA5')
    plt.plot([0, 1], [.6,.6] , 'k--')
    plt.plot([.6,.6], [0,1] , 'k--')

    # plt.plot([0, .45], [0, .45], '-', lw=3, color="#E17D0E")
    plt.annotate("", xy=(0.45, 0.45), xytext=(0, 0),arrowprops=dict(arrowstyle="<->", color="#DD361F", lw=2.5))
    plt.text(.0, .05, "Outlier-ness", fontsize=14, fontweight='bold', rotation=34, color='k')

    plt.xlim(-.02,1.02)
    plt.ylim(-.02,1.02)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.xlabel("Mean Error", fontsize=14, fontweight='bold')
    plt.ylabel("S. dev of the Error", fontsize=14, fontweight='bold')
    plt.savefig(save_loc)

def plot_olness_dist(enpls=None, save_loc="./images/unsup/olness_th.png"):
    
    fig, ax = plt.subplots()
    N, bins, patches = ax.hist(enpls['olness'], bins=50, color='#DD361F')
    for i in range(0,6):
        patches[i].set_facecolor('#1C6EA5')
    for i in range(6, len(patches)):
        patches[i].set_facecolor('#DD361F')
    plt.plot([enpls['olness'].quantile(.84), enpls['olness'].quantile(.84)], [0,max(N)+100], 'k--')

    plt.ylim(0, max(N)+100)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.ylabel('Number of molecules',fontsize=14, fontweight='bold')
    plt.xlabel('Outlier-ness',fontsize=14, fontweight='bold')
    plt.savefig(save_loc)


# def process_detect_results(df, save_res=False):
    
#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{1}.pkl', 'rb') as f:
#         f1 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{2}.pkl', 'rb') as f:
#         f2 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{3}.pkl', 'rb') as f:
#         f3 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{4}.pkl', 'rb') as f:
#         f4 = pickle.load(f)

#     f={}
#     for i in df.smiles.values:
#         all_v = f1[i]+f2[i]+f3[i]+f4[i]
#         f[i] = all_v

#     res=[]
#     for k in f:

#         abs_err = np.abs(f[k])
#         res.append( [k, np.mean(abs_err), np.std(abs_err), len(f[k]) ] )

#     res = pd.DataFrame(res, columns=['smiles', 'means', 'sdev', 'ntimes'])


#     enpls = res
#     enpls['means_sc'] = (enpls['means'] - enpls['means'].min())/(enpls['means'].max() - enpls['means'].min())
#     enpls['sdev_sc'] = (enpls['sdev'] - enpls['sdev'].min())/(enpls['sdev'].max()-enpls['sdev'].min())
#     enpls['olness'] = np.sqrt(enpls['means_sc']**2 + enpls['sdev_sc']**2)
#     enpls['olness_nonsc'] = np.sqrt(enpls['means']**2 + enpls['sdev']**2)
    
#     if save_res:
#         enpls.to_csv(f"{config.DETECT_DIR}/enpls.csv", index=False)
#     else:
#         enpls = pd.read_csv(f"{config.DETECT_DIR}/enpls.csv")
#     # enpls.to_csv("./enpls_for_test.csv", index=False)
    
#     return enpls


# def process_detect_results(df, save_res=False):
    
#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{1}.pkl', 'rb') as f:
#         f1 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{2}.pkl', 'rb') as f:
#         f2 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{3}.pkl', 'rb') as f:
#         f3 = pickle.load(f)

#     with open(f'{config.DETECT_DIR}/smiles_error_sup_et_{4}.pkl', 'rb') as f:
#         f4 = pickle.load(f)

#     f={}
#     for i in df.smiles.values:
#         all_v = f1[i]+f2[i]+f3[i]+f4[i]
#         f[i] = all_v

#     res=[]
#     for k in f:

#         abs_err = np.abs(f[k])
#         res.append( [k, np.mean(abs_err), np.std(abs_err), len(f[k]) ] )

#     res = pd.DataFrame(res, columns=['smiles', 'means', 'sdev', 'ntimes'])


#     enpls = res
#     enpls['means_sc'] = (enpls['means'] - enpls['means'].min())/(enpls['means'].max() - enpls['means'].min())
#     enpls['sdev_sc'] = (enpls['sdev'] - enpls['sdev'].min())/(enpls['sdev'].max()-enpls['sdev'].min())
#     enpls['olness'] = np.sqrt(enpls['means_sc']**2 + enpls['sdev_sc']**2)
#     enpls['olness_nonsc'] = np.sqrt(enpls['means']**2 + enpls['sdev']**2)
    
#     if save_res:
#         enpls.to_csv(f"{config.DETECT_DIR}/enpls.csv", index=False)
#     else:
#         enpls = pd.read_csv(f"{config.DETECT_DIR}/enpls.csv")
#     # enpls.to_csv("./enpls_for_test.csv", index=False)
    
#     return enpls

