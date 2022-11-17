import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
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

def get_transformed_data(train,val,test, features, y):

    x_train = train.loc[:,features].values
    x_val = val.loc[:,features].values
    x_test = test.loc[:,features].values

    y_train = train[y].values
    y_val = val[y].values
    y_test = test[y].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    
    return x_train,y_train, x_test, y_test, x_val, y_val, scaler



def run_ol(part, calcs, df, features, save_path):
    
    print("outlier detection using extratrees regression")
 
    smiles_error = {}
    for i in df.smiles:
        smiles_error[i] = []

    for i in tqdm(range(calcs)):
        trainx, valx = train_test_split(df, test_size = .2, shuffle=True, random_state=random.randint(0,20000))
        testx, valx = train_test_split(valx, test_size = .5, shuffle=True, random_state=random.randint(0,20000))


        x_train, y_train, x_test, y_test, x_val, y_val, sc = get_transformed_data(train   = trainx, 
                                                                             val     = valx, 
                                                                             test    = testx, 
                                                                             features = features, 
                                                                             y       = "log_sol")

        reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(x_train, y_train)
        pred_test = reg.predict(x_test).ravel()


        error = y_test - pred_test

        for ip, test_smiles in enumerate(testx.smiles.values):
            smiles_error[test_smiles].append( error[ip] )


    # with open('smiles_error_1000.pkl', 'wb') as f:
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
    with open(f'{save_path}/smiles_error_sup_et_{part}.pkl', 'wb') as f:
        pickle.dump(smiles_error, f)


# after running the models
def find_olpercent(data_path, 
                    res_path, 
                    target_name='log_sol',
                     save_res=False, 
                     rerun_threshold_detection=False,
                     n_parts=1):
    
    """
    This function is specific to process_detect_results function above.
    """
    
    # train = pd.read_csv(f"{config.DATA_DIR}/sets/train.csv")
    # val = pd.read_csv(f"{config.DATA_DIR}/sets/val.csv")
    # df = pd.concat([train,val], axis=0).reset_index(drop=True)
    df = pd.read_csv(data_path)
    features = df.drop(['smiles', target_name], axis=1).columns.values


    
    if rerun_threshold_detection:

        stats, enpls = collect_prediction_results(df, res_path, save=save_res, nparts=n_parts)
        ols_with_olpercent(df, enpls, save_folder=res_path)
        olp_versus_r2 = pd.read_csv(f"{res_path}/cv_results.csv")
        print("ols_with_olpercent has to run using job script")

    else:
        enpls = process_detect_results(df, save_res=False)    
        olp_versus_r2 = pd.read_csv(f"{res_path}/cv_results.csv")
        
        
    return olp_versus_r2, enpls, df




def ols_with_olpercent(df, enpls, save_folder):
    
    save_folder = os.path.join(save_folder, 'ol_thresh_res')
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    cv_results = []
    for ie, i in enumerate(range(95,60,-5)):

        df_ref = df.copy()

        percent = i/100
        print("percent: ", percent)
        olness_th = enpls['olness'].quantile(percent)


        smiles_ol = enpls.loc[enpls.olness > olness_th, : ]['smiles'].values


        df_ol = df_ref[df_ref.smiles.isin(smiles_ol)]
        df_nol = df_ref[~df_ref.smiles.isin(smiles_ol)]


        res = get_cv(df_nol)
        ol_percent = 100*df_ol.shape[0]/df_ref.shape[0]

        
        cv_results.append([res.r2.mean(), res.rmse.mean(), ol_percent, ie])
        
        df_ol.to_csv(f'{save_folder}/outliers_olness_p{ie}.csv', index=False)
        df_nol.to_csv(f'{save_folder}/non_outliers_olness_p{ie}.csv', index=False)
        

    pd.DataFrame(cv_results, columns=['r2', 'rmse', 'olp', 'prolp']).to_csv(f'{save_folder}/cv_results.csv', index=False)



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

def get_cv(dft):
    
    df = dft.copy()
    df = df.sample(frac=1).reset_index(drop=True)
#     df.reset_index(drop=True, inplace=True)
    to_remove = ['smiles','log_sol']
    
    model = ExtraTreesRegressor(n_estimators=100)
    kf = KFold(n_splits=5)

    res=[]
    for train_index, test_index in kf.split(df):
#         print("TRAIN:", train_index, "TEST:", test_index)
        df_train, df_test = df.loc[train_index,:], df.loc[test_index, :]

        x_train = df_train.drop(to_remove, axis=1)
        x_test = df_test.drop(to_remove, axis=1)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.values)
        x_test = scaler.transform(x_test.values)

        y_train = df_train.log_sol.values
        y_test = df_test.log_sol.values

        # For training, fit() is used
        model.fit(x_train, y_train)

        # Default metric is R2 for regression, which can be accessed by score()
        pred = model.predict(x_test)

        # For other metrics, we need the predictions of the model


        rmse = mean_squared_error(y_pred = pred, y_true = y_test)**.5
        r2 = r2_score(y_pred = pred, y_true = y_test)
        res.append([rmse, r2])
        
    res = pd.DataFrame(res, columns=['rmse', 'r2'])
    
    return res

