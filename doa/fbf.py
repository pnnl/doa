import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# import config


class fbf_results():
    def __init__(self, df_ref, df_train, df_test, save_path):
        
        self.df_ref = df_ref
        self.df_train = df_train
        self.df_test = df_test
        self.to_remove = ['smiles','log_sol']
        self.save_path = save_path

    def get_model_scaler(self):
        

        model = ExtraTreesRegressor(n_estimators=100, random_state=42)


        x_train = self.df_train.drop(self.to_remove, axis=1)
        features = x_train.columns.tolist()
        # x_test = df_test.drop(to_remove, axis=1)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.values)
        # x_test = scaler.transform(x_test.values)

        y_train = self.df_train.log_sol.values
        # y_test = df_test.log_sol.values

        # For training, fit() is used
        model.fit(x_train, y_train)
        # Default metric is R2 for regression, which can be accessed by score()
        # pred = model.predict(x_test)

        # For other metrics, we need the predictions of the model
        # rmse = mean_squared_error(y_pred = pred, y_true = y_test)**.5
        # r2 = r2_score(y_pred = pred, y_true = y_test)
        
        self.model = model
        self.scaler = scaler

        return model, scaler, features


    def get_doa_results(self, doa, olp_th=None, rmse_th=None, r2_th=None, remove_from_train=False):

        all_outsmiles = []
        ol_test_smiles = []
        ol_train_smiles = []
        nol_test = []
        results=[]
        saved = 0
        
        n_test_mols = self.df_test.shape[0]
        
        for i in tqdm(doa.index):
            prop = doa.loc[i, 'prop']
            direction = doa.loc[i, 'amax']
            direction2 = doa.loc[i, 'direction']
            th = doa.loc[i, 'th']


            df_train = self.df_train.copy()
            df_test = self.df_test.copy()

            # print(prop, th)

            if direction == 1:
                all_outsmiles.extend(self.df_ref[self.df_ref[prop] > th].smiles.values.tolist())
                ol_test_smiles.extend(df_test[df_test[prop] > th].smiles.values.tolist())
                ol_train_smiles.extend(df_train[df_train[prop] > th].smiles.values.tolist())


            elif direction == 0:
                all_outsmiles.extend(self.df_ref[self.df_ref[prop] < th].smiles.values.tolist())
                ol_test_smiles.extend(df_test[df_test[prop] < th].smiles.values.tolist())
                ol_train_smiles.extend(df_train[df_train[prop] < th].smiles.values.tolist())


#             # this may not be necessary
#             ol_test_smiles = list(set(ol_test_smiles))
#             ol_test_smiles = list(set(ol_test_smiles))
            
            if remove_from_train:
                print('removing from train...')
                df_train = df_train[~df_train.smiles.isin(ol_train_smiles)]
                x_train = df_train.drop(self.to_remove, axis=1)
                features = x_train.columns.tolist()
                
                self.model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
                x_train = self.scaler.fit_transform(x_train.values)
                y_train = df_train.log_sol.values
                self.model.fit(x_train, y_train)

            df_test = df_test[~df_test.smiles.isin(ol_test_smiles)]


            x_test = df_test.drop(self.to_remove, axis=1)
            x_test = self.scaler.transform(x_test.values)
            y_test = df_test.log_sol.values

            pred = self.model.predict(x_test)

            rmse = mean_squared_error(y_pred = pred, y_true = y_test)**.5
            r2 = r2_score(y_pred = pred, y_true = y_test)


#             unq_all_ols = list(set(all_outsmiles))
            unq_all_ols = list(set(ol_train_smiles + ol_test_smiles))
            olp = len(unq_all_ols)/self.df_ref.shape[0]
            test_olp = len(list(set(ol_test_smiles)))/n_test_mols
            n_test_ols = len(list(set(ol_test_smiles)))
            n_all_ols = len(unq_all_ols)
            n_train_ols = len(list(set(ol_train_smiles)))

            results.append([rmse, r2, prop, direction, direction2, th, i, test_olp, n_test_ols, n_train_ols, n_all_ols, olp ])

            
            if rmse_th and rmse < rmse_th:
                break
            elif r2_th and r2 > r2_th:
                print('checking r2....')
                break
            elif olp_th and olp > olp_th:
                break

            # break


        results = pd.DataFrame(results, columns=['rmse', 'r2', 'prop', 'amax','direction2', 'th', 'i',\
                                                 'test_olp', 'n_test_ols', 'n_train_ols', 'n_all_ols', 'all_olp'])
        self.doa_res = results
        
        if self.save_path:
            with open(f"{self.save_path}/all_outsmiles.pkl", "wb") as f:
                pickle.dump(all_outsmiles, f)

            with open(f"{self.save_path}/ol_test_smiles.pkl", "wb") as f:
                pickle.dump(ol_test_smiles, f)

            results.to_csv(f"{self.save_path}/doa_results.csv", index=False)

        return results, all_outsmiles, ol_test_smiles
    
    
    
    
    def get_random_mol_removal_results(self, doa_res, remove_from_train=False):
    
        results_random=[]
        for i in doa_res.index:

            df_test = self.df_test.copy()
            df_train = self.df_train.copy()
            df_combined = pd.concat([df_test, df_train], axis=0)
            df_combined.reset_index(drop=True, inplace=True)
            
            prop = doa_res.loc[i, 'prop']
            n_all_ols = doa_res.loc[i, 'n_all_ols']
            n_test_ols = doa_res.loc[i, 'n_test_ols']
            if remove_from_train:
                print("removing from train...")
                sample = df_combined.sample(n=n_all_ols)
                df_train = df_train[~df_train.smiles.isin(sample.smiles.values)]
                x_train = df_train.drop(self.to_remove, axis=1)
                features = x_train.columns.tolist()
                
                self.model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
                x_train = self.scaler.fit_transform(x_train.values)
                y_train = df_train.log_sol.values
                self.model.fit(x_train, y_train)    
            else:
                sample = df_test.sample(n=n_test_ols)
            
            df_test = df_test[~df_test.smiles.isin(sample.smiles.values)]

            x_test = df_test.drop(self.to_remove, axis=1)
            x_test = self.scaler.transform(x_test.values)
            y_test = df_test.log_sol.values


            pred = self.model.predict(x_test)

            rmse = mean_squared_error(y_pred = pred, y_true = y_test)**.5
            r2 = r2_score(y_pred = pred, y_true = y_test)


            results_random.append([rmse, r2, prop, i, n_test_ols, n_all_ols, n_all_ols/self.df_ref.shape[0] ])

        results_random = pd.DataFrame(results_random, columns=['rmse', 'r2', 'prop', 'i', 'n_test_ols',
                                                              'n_all_ols', 'all_olp'])
        self.random_res = results_random
        
        results_random.to_csv(f"{self.save_path}/random_doa_results.csv", index=False)

        return results_random
    
    
    def get_random_feature_removal_results(self, doa_data, remove_from_train=False):
    
        doa_table = doa_data.copy()
        feature_list = doa_table.prop.values.tolist()
        doa_table.set_index('prop', inplace=True)
        n_steps = self.doa_res.shape[0]
        
        results_random_features = []
        ol_test_smiles = []
        ol_train_smiles = []
        for i in tqdm(range( n_steps )):
                      
            prop = random.sample(feature_list, 1)[0]
            direction = doa_table.loc[prop, 'amax']
            th = doa_table.loc[prop, 'th']
            
        
            df_test = self.df_test.copy()
            df_train = self.df_train.copy()

            if direction == 1:
                ol_test_smiles.extend(df_test[df_test[prop] > th].smiles.values.tolist())
                ol_train_smiles.extend(df_train[df_train[prop] > th].smiles.values.tolist())
            elif direction == 0:
                ol_test_smiles.extend(df_test[df_test[prop] < th].smiles.values.tolist())
                ol_train_smiles.extend(df_train[df_train[prop] < th].smiles.values.tolist())
    
            if remove_from_train:
                print('removing from train...')
                df_train = df_train[~df_train.smiles.isin(ol_train_smiles)]
                x_train = df_train.drop(self.to_remove, axis=1)
                features = x_train.columns.tolist()
                
                self.model = ExtraTreesRegressor(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
                x_train = self.scaler.fit_transform(x_train.values)
                y_train = df_train.log_sol.values
                self.model.fit(x_train, y_train)    
                
            ol_test_smiles = list(set(ol_test_smiles))
            n_test_ols = len(ol_test_smiles)

            df_test = df_test[~df_test.smiles.isin(ol_test_smiles)]

            x_test = df_test.drop(self.to_remove, axis=1)
            x_test = self.scaler.transform(x_test.values)
            y_test = df_test.log_sol.values


            pred = self.model.predict(x_test)

            rmse = mean_squared_error(y_pred = pred, y_true = y_test)**.5
            r2 = r2_score(y_pred = pred, y_true = y_test)

            unq_all_ols = list(set(ol_train_smiles + ol_test_smiles))
            all_olp = len(unq_all_ols)/self.df_ref.shape[0]

            results_random_features.append([rmse, r2, prop, i, n_test_ols, unq_all_ols, all_olp ])

        results_random_features = pd.DataFrame(results_random_features, columns=['rmse', 'r2', 'prop', 'i',
                                                                                 'n_test_ols', 'n_all_ols', 'all_olp' ])
        results_random_features.to_csv(f"{self.save_path}/results_random_features.csv", index=False)
        
        return results_random_features


    def bin_rand_feature_removal(self, results_random_features):

        num_runs = len(results_random_features)
        bin_width = 50

        max_rf = max([results_random_features[i].nout.max() for i in range( num_runs )])

        res={'r2':{ f'{l}:{l+bin_width}':[] for l in range(0,max_rf, bin_width)  },
             'rmse':{ f'{l}:{l+bin_width}':[] for l in range(0,max_rf, bin_width)  }
            }
        for j in range(5):
            tmp = results_random_features[j]
            for i in range(0,max_rf, bin_width):
                low = i
                high = i + bin_width
                # print(low, high)
                tmp_sub = tmp[np.logical_and((tmp.nout > low) ,  (tmp.nout <= high))]

                r2_vals = tmp_sub['r2'].values.tolist()
                rmse_vals = tmp_sub['rmse'].values.tolist()
                res['r2'][ f'{low}:{high}' ].extend(r2_vals)
                res['rmse'][ f'{low}:{high}' ].extend(rmse_vals)

        # res['r2']

#         rand_features_r2=[]
#         for k,v in res['r2'].items():
#             k.split(':')[0]
#             bin_mean=np.mean([int(k.split(':')[0]), int(k.split(':')[1])])
#             vmean = np.mean(v)
#             vsdev = np.std(v)
#             rand_features_r2.append([bin_mean, vmean, vsdev])

#         rand_features_rmse=[]
#         for k,v in res['rmse'].items():
#             k.split(':')[0]
#             bin_mean=np.mean([int(k.split(':')[0]), int(k.split(':')[1])])
#             vmean = np.mean(v)
#             vsdev = np.std(v)
#             rand_features_rmse.append([bin_mean, vmean, vsdev])
        rand_features_rmse = self.get_bin_metric_means(res, 'rmse')
        rand_features_r2 = self.get_bin_metric_means(res, 'r2')

        rand_features_rmse = pd.DataFrame(rand_features_rmse, columns=['nout', 'rmse', 'rmse_sdev'])
        rand_features_r2 = pd.DataFrame(rand_features_r2, columns=['nout', 'r2', 'r2_sdev'])

        rand_features = pd.merge(rand_features_r2, rand_features_rmse, on='nout')

        rand_features.dropna(axis=0, inplace=True)

        return rand_features


    def n_random_feature_removal(self, doa_olness=None, remove_from_train=False):
    
        results_random_features = \
        [self.get_random_feature_removal_results(doa_data=doa_olness, remove_from_train=remove_from_train) for _ in range(5)]

        with open(f"{self.save_path}/results_random_features.pkl", 'wb') as f:
            pickle.dump(results_random_features, f)

        res_rand_features = self.bin_rand_feature_removal(results_random_features)
        res_rand_features.to_csv(f'{self.save_path}/res_rand_features.csv', index=False)
        return res_rand_features

    
    
    def get_bin_metric_means(self, res, metric='r2'):
    
        rand_features =[]
        for k,v in res[metric].items():
            k.split(':')[0]
            bin_mean=np.mean([int(k.split(':')[0]), int(k.split(':')[1])])

            if len(v) == 0:
                continue

            vmean = np.mean(v)
            vsdev = np.std(v)
            rand_features.append([bin_mean, vmean, vsdev])
        
        return rand_features
    
    def n_random_mol_removal(self, doa_results=None, remove_from_train=False):
    
        random_doa_results=[]
        for _ in range(5):
            random_doa_results.append(self.get_random_mol_removal_results(doa_res=doa_results, remove_from_train=remove_from_train))

        with open(f"{self.save_path}/random_doa_results.pkl", "wb") as f:
            pickle.dump(random_doa_results, f)

        rand_rmse = pd.concat([i[['rmse']] for i in random_doa_results],  axis=1)
        rmse_mean = rand_rmse.mean(axis=1)
        rmse_sdev = rand_rmse.std(axis=1)

        return random_doa_results, rmse_mean, rmse_sdev
