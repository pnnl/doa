import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import config
from sklearn.model_selection import train_test_split
from joblib import dump, load
from doa.fbf import fbf_results
from sklearn.metrics import mean_squared_error, r2_score
import pickle


class OLClassify:
    
    def __init__(self, df_il, df_ol):
        
        self.df_ol = df_ol.copy()
        self.df_il = df_il.copy()
     

        self.df_ol.loc[:,'label'] = 1
        self.df_il.loc[:,'label'] = 0


        self.dff = pd.concat([self.df_il, self.df_ol], axis=0)
        self.dff.reset_index(drop=True, inplace=True)
        
        self.X = self.dff.drop(['smiles', 'log_sol', 'label'], axis=1)
        self.features = self.X.columns.tolist()
        self.X = self.X.values
        self.y = self.dff['label'].values
        
        
    def fit(self, random_state):
        
        sc = StandardScaler()
        model = ExtraTreesClassifier(random_state=random_state)
        X = sc.fit_transform(self.X)
        model.fit(X, self.y)
        
        return model, sc, self.features
        

            
    def kfold_predict(self, n=5):
        
        skf = StratifiedKFold(n_splits=n)
        skf.get_n_splits(self.X, self.y)

#         print(skf)

        cls_dicts = []
        models= []
        predictions=[]
        for train_index, test_index in skf.split(self.X, self.y):
#             print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
        
            sc = StandardScaler()
            
            X_train = sc.fit_transform(X_train)            
            X_test = sc.transform(X_test)
            
            et = ExtraTreesClassifier()
            et.fit(X_train, y_train)
            models.append(et)
            
            
            pred = et.predict(X_test)
#             print(np.unique(pred))
#             cls_dict = pd.DataFrame(classification_report(y_pred=pred, y_true=y_test, output_dict=True))
            cls_dict = classification_report(y_pred=pred, y_true=y_test, output_dict=True)
            cls_dicts.append(cls_dict)
            predictions.append({'true':y_test, 'pred':pred})
        
            

            
#         f1_1 = [a.loc['f1-score', '1'] for a in cls_dicts]
#         f1_0 = [a.loc['f1-score', '0'] for a in cls_dicts]

        def get_means(metric='f1-score'):
        
            cls_1 = [a['1'][metric] for a in cls_dicts]
            cls_0 = [a['0'][metric] for a in cls_dicts]

            cls_1_mean, cls_1_sdev = np.mean(cls_1), np.std(cls_1) # outlier        
            cls_0_mean, cls_0_sdev = np.mean(cls_0), np.std(cls_0) # inlier

            return cls_1_mean, cls_1_sdev, cls_0_mean, cls_0_sdev
        
    
        res = {'f1s': get_means(metric='f1-score'),
            'pr':get_means(metric='precision'),
            'rc': get_means(metric='recall')}
        
#         'precision': 1.0, 'recall'

        return cls_dicts, res, models, predictions
    
    
    
    
class OLClassifyResults:
    def __init__(self, train, test, df_nols, df_ref, save_path):
        self.train = train
        self.test = test
        self.df_nols = df_nols
        self.df_ref = df_ref
        self.save_path = save_path
        
    def fbf_using_clsf(self, oltype_name, ols, random_state=42, calc_id=0):
        
        olname = oltype_name
        df_ols = ols

        clf_obj = OLClassify(self.df_nols, df_ols)
        model1, sc1, features1 = clf_obj.fit(random_state)

        pickle.dump(model1, open(f'{self.save_path}/model_{olname}_{calc_id}.pkl', 'wb'))
        pickle.dump(sc1, open(f'{self.save_path}/sc_{olname}_{calc_id}.pkl', 'wb'))

        test_x = self.test.loc[:, features1]
        test_x = sc1.transform(test_x.values)
        test_pred = model1.predict(test_x)
        test_pred_prob = model1.predict_proba(test_x)

        test_olness = self.test.copy()
        test_olness.loc[:, 'clf_olness'] = test_pred_prob[:, 1]
        test_olness.loc[:, 'clf_nolness'] = test_pred_prob[:, 0]
        test_olness.sort_values(by='clf_olness', ascending=False, inplace=True)
        test_olness.reset_index(drop=True, inplace=True)
        test_olness.to_csv(f'{self.save_path}/test_olness_{olname}_{calc_id}.csv', index=False)

        # test_olness
        doa_res = pd.read_csv(f'{config.DOA_DIR}/doa_results.csv')
        fbf_obj = fbf_results(df_ref = self.df_ref, df_train = self.train, df_test=self.test, save_path=None)
        # train a model using train set
        model2, scaler2, features2 = fbf_obj.get_model_scaler()


        results=[]
        for i in doa_res.index:
            n_test_ols = doa_res.loc[i, 'n_test_ols']
            pred_nols = test_olness.iloc[n_test_ols:,:]

            test_x = pred_nols.loc[:, features2]
            test_x = scaler2.transform(test_x.values)
            test_y = pred_nols.log_sol.values

            pred = model2.predict(test_x)


            rmse = mean_squared_error(y_pred = pred, y_true = test_y)**.5
            r2 = r2_score(y_pred = pred, y_true = test_y)

            results.append([rmse, r2, n_test_ols])

        results = pd.DataFrame(results, columns=['rmse', 'r2', 'nout'])
        results.to_csv(f'{self.save_path}/clsf_fbf_res_{olname}_{calc_id}.csv', index=False)
        return results
    
    