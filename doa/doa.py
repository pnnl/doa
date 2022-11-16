import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns
# import config
# plt.ioff()

def get_transformed_data(train,val,test, features, y):

    """
    standard scale the features and
    return X and y.
    """
    x_train = train.loc[:, features].values
    x_val = val.loc[:, features].values
    x_test = test.loc[:, features].values

    y_train = train[y].values
    y_val = val[y].values
    y_test = test[y].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train,y_train, x_test, y_test, x_val, y_val, scaler


def scale_oldf(df_ref, df_ol, df_nol, cols):

    df_sc = df_ref.copy()
    sc = StandardScaler()

    # standard scale the entire dataset
    df_sc.loc[:, cols] = sc.fit_transform(df_sc.loc[:, cols])

    # create the ol dataframe with scaled features
    df_ol = df_sc[df_sc.smiles.isin(df_ol.smiles.values)]
    df_ol.reset_index(drop=True, inplace=True)

    # create the nol dataframe with scaled features
    df_nol = df_sc[df_sc.smiles.isin(df_nol.smiles.values)]
    df_nol.reset_index(drop=True, inplace=True)

    return df_ol, df_nol

def get_range_diffs(df_ol, df_nol, cols):

    res2, res2_all, diff_not_found  = [], [], []
    for prop in tqdm(cols):
        # make sure that the features are float type
        df_ol[prop] = df_ol[prop].astype(float)
        df_nol[prop] = df_nol[prop].astype(float)

    #     argmax = np.argmax([ abs(df_ol[prop].min() - df_nol[prop].min()),
    #                         abs(df_ol[prop].max() - df_nol[prop].max()) ])

        # is the difference on the negative side or possitive side?
        # find the differences between max values and min values
        # of the ol and nol distributions.
        # => this calcularion assumes that outlier distribution is
        # wider than the inlier distribution
        pos_side_diff = df_ol[prop].max() - df_nol[prop].max()
        neg_side_diff =  df_nol[prop].min() - df_ol[prop].min()
        # which side's difference is larger?
        argmax = np.argmax([ abs(neg_side_diff), abs(pos_side_diff) ])

    #         argmax = np.argmax([ abs(df_ol[prop].min() - df_nol[prop].min()),
    #                             abs(df_ol[prop].max() - df_nol[prop].max()) ])


        # if on negative side, label 0
        if (argmax == 0) and (neg_side_diff>0):
            res2.append([prop, neg_side_diff, 0 ])
            res2_all.append([prop, abs(pos_side_diff), 1]) # just finding the magnitude of the difference,
            res2_all.append([prop, abs(neg_side_diff), 0]) # want to know which one is larger

        # if on positive side, label 1
        elif (argmax == 1) and (pos_side_diff)>0:
            res2.append([prop, pos_side_diff, 1 ])
            res2_all.append([prop, abs(pos_side_diff), 1])
            res2_all.append([prop, abs(neg_side_diff), 0])            

        else:
            # print(prop, argmax, pos_side_diff, neg_side_diff)
            diff_not_found.append([prop, argmax, pos_side_diff, neg_side_diff])
            
    diff_not_found = pd.DataFrame(diff_not_found, columns=['prop', 'argmax', 'pos_side_diff', 'neg_side_diff'])


    res2 = pd.DataFrame(res2, columns = ['feature', 'diffrnc', 'amax'])
    res2 = res2.sort_values(by='diffrnc', ascending=False)
    res2.reset_index(drop=True, inplace=True)
    
    res2_all = pd.DataFrame(res2_all, columns = ['feature', 'diffrnc', 'amax'])
    res2_all = res2_all.sort_values(by='diffrnc', ascending=False)
    res2_all.reset_index(drop=True, inplace=True)
    res2_all = res2_all.drop_duplicates(subset=['feature'], keep='first')
    # res2_all.sort_values(by=['diffrnc'], ascending=False).reset_index(drop=True, inplace=True)
    res2_all.reset_index(drop=True, inplace=True)
    
    # print(res2_all)
    
    return res2, res2_all, diff_not_found

def get_thresholds(res2, df_ref):

    # res2 is the output of get_range_diffs
    # it should be sorted in the descending order
    # todo: test whether this is sorted properly
    prop_th2 = []

    print("using IQR from definition")
    for i in tqdm(res2.index):
        prop = res2.loc[i, 'feature']
        amax = res2.loc[i, 'amax']
        diff = res2.loc[i, 'diffrnc']

        # TODO: test the part below using this
        q75, q25 = np.percentile(df_ref[prop].values, [75 ,25])
        pos_th = q75 + (q75-q25)*1.5
        neg_th = q25 - (q75-q25)*1.5
    
#         B = plt.boxplot(df_ref[prop].values);
#         ol = [item.get_ydata()[1] for item in B['whiskers']]
        ol = [neg_th, pos_th]
        
        if amax == 1:
            th = ol[1]
            prop_th2.append([prop, th, amax, diff, pos_th])
        elif amax == 0:
            th = ol[0]
            prop_th2.append([prop, th, amax, diff, neg_th])


    prop_th2 = pd.DataFrame(prop_th2, columns = ['prop','th', 'amax', 'diffrnc', 'test_th'])

    return prop_th2




# MAIN class to find OLs
class find_th():

#     to_remove = ['cas', 'temp', 'inchi', 'ref']
    to_drop=['smiles', 'log_sol']

    def __init__(self, ol, nol, all_data, enpls):

        self.df_ref_all = all_data.copy()
        self.enpls = enpls.copy()
#         data_dir = data_dir
#         data_dir="~/solubility_prediction/property_prediction/data/pnnl1.92/data/"

#         self.train = pd.read_csv(data_dir+"/train.csv")
#         self.val = pd.read_csv(data_dir+"/val.csv")
#         self.test = pd.read_csv(data_dir+"/test.csv")

#         self.train.drop(find_th.to_remove, axis=1, inplace=True)
#         self.val.drop(find_th.to_remove, axis=1, inplace=True)
#         self.test.drop(find_th.to_remove, axis=1, inplace=True)

        self.cols = self.df_ref_all.columns.tolist()
        self.cols.remove('smiles')
        self.cols.remove('log_sol')
        
        self.df_ref_all.loc[:, self.cols] = self.df_ref_all.loc[:, self.cols].astype(float)

#         df_ref = pd.concat([self.train, self.val])
#         self.df_ref_all = pd.concat([self.train, self.val, self.test])
#         self.df_ref = df_ref.reset_index(drop=True)

        self.ol = ol
        self.nol = nol

        # get scaled features; using all the features
        self.df_olSC, self.df_nolSC = scale_oldf(self.df_ref_all, self.ol, self.nol, self.cols)

        # get the differences in ol nol distributions
        self.res2, self.res2_all, self.diff_not_found = get_range_diffs(self.df_olSC, self.df_nolSC, self.cols)

        # get the thresholds using all the data (train,val,test)
        # self.prop_th = get_thresholds(self.res2, self.df_ref_all)
        self.prop_th = get_thresholds(self.res2_all, self.df_ref_all)


    def remove_corr_from_propth(self, th):
        # remomve highly correlated features
        # this just changes the features we use to
        # define the domain of applicability
        df = self.df_ref_all.copy()
        df = df.drop(['smiles', 'log_sol'], axis=1)

        corr = df.corr()
        cols = self.prop_th.prop.values.tolist()

        hc_pairs, hc_todrop = [], []
        for i in tqdm(range(len(cols))):
            for j in range(i+1, len(cols) ):
        #         j=0
                cv = corr[ cols[i] ][ cols[j] ]
        #         print(high_corr[i],  high_corr[j],  cv)
                if abs(cv) > th:
                    hc_pairs.append([cols[i], cols[j] ])
                    hc_todrop.append(cols[j])

        prop_th_new = self.prop_th[~self.prop_th.prop.isin(hc_todrop)]
        prop_th_new.reset_index(drop = True, inplace = True)

        self.prop_th_new = prop_th_new
        return self.prop_th_new
    
    
    def get_doa_table(self):
        
        thresholds = self.prop_th_new.copy()

        for i in thresholds.index:
            am = thresholds.loc[i, 'amax']
            thresh  = thresholds.loc[i, 'th']
            # if am == 1, indomain molecules should have property values less than the threshold
            if am ==1:
                s = "<="
            # if am == 0, indomain molecules should have property values grater than the threshold
            elif am == 0:
                s = ">="

            if self.df_ref_all[ thresholds.loc[i, 'prop'] ].min() == thresh:
                s = "=="
            thresholds.loc[i, 'direction'] = s

        # th['direction'] = th['amax'].apply(lambda x: "<=" if x==1 else ">=")
        thresholds['descriptor'] = thresholds['prop']
        thresholds['threshold'] = thresholds['th']

        for i in thresholds.index:
            prop = thresholds.loc[i, 'prop']
            am = thresholds.loc[i, 'amax']

            if am == 1:
                nol_row = self.df_ref_all[self.df_ref_all[prop] > thresholds.loc[i,'th']].shape[0]

            elif am == 0:
                nol_row = self.df_ref_all[self.df_ref_all[prop] < thresholds.loc[i,'th']].shape[0]

            # print(nol_row)
            thresholds.loc[i, '#ols'] = nol_row

        thresholds['#ols'] = thresholds['#ols'].astype(int)
        # os.mkdir('doa')
        # th.to_csv("./doa.csv", index=False)
        doa = thresholds.copy()
        self.doa = doa
        
        self.doa_olness = self.doa_by_olness(doa)
        
        return doa, self.doa_olness
    
    
    def doa_by_olness(self, doa):
        for i in tqdm(doa.index):
            prop = doa.loc[i, 'prop']
            thresh = doa.loc[i, 'th']
            direc = doa.loc[i, 'amax']

            if direc == 1:
                sms = self.df_ref_all[self.df_ref_all[prop] > thresh]['smiles'].values
            elif direc == 0:
                sms = self.df_ref_all[self.df_ref_all[prop] < thresh]['smiles'].values

            mean_oln = self.enpls[self.enpls.smiles.isin(sms)]['olness'].mean()
            doa.loc[i, 'mean_oln'] = mean_oln

        doa_olness  = doa.sort_values(by='mean_oln', ascending=False)
        doa_olness.reset_index(drop=True, inplace=True)
        return doa_olness
        
    

    
    
    def th_plot(self, corr_th, figsize, sns_fs):
        # corr_th = threshold to remove highly
        # correlated features
        rows, columns = figsize
        n_plots = int(rows * columns)
        
        sns.set(font_scale = sns_fs)
        
        prop_th_new = self.remove_corr_from_propth(corr_th)
        df = self.df_ref_all.copy()
        df.loc[df.smiles.isin( self.ol.smiles).values, 'type'] = 0
        df.loc[df.smiles.isin( self.nol.smiles).values, 'type'] = 1

        # Show the distributions of outliers and non-outliers
        fig, ax = plt.subplots(rows, columns, figsize=(22,12))
        axes = ax.ravel()
        for ie, prop in enumerate(self.prop_th_new.prop.values[:n_plots]):
            g = sns.boxplot(x='type', y=prop, data=df, ax=axes[ie])

            g.set_xticks(range(2)) # <-- set the ticks first
            g.set_xticklabels(['Outliers','Non-outliers'], fontsize=16)

            axes[ie].set_xlabel("")
        plt.fig("./images/ol_nol_diff.png")


        
    def save_data(self, save_path):
        self.df_ref_all.to_csv(f"{save_path}/all_data.csv", index=False)
        self.doa.to_csv(f"{save_path}/doa.csv", index=False)
        self.doa_olness.to_csv(f"{save_path}/doa_olness.csv", index=False)
        
    @classmethod
    def get_nols(df, prop_th, olp_th):

        total = df.shape[0]

        all_ol_smiles=[]
        for i in tqdm(prop_th.index):
            prop = prop_th.loc[i, 'prop']
            th = prop_th.loc[i, 'th']
            amax = prop_th.loc[i, 'amax']


            if amax == 1:
                ol_smiles = df[df[prop] > th]['smiles'].values
            elif amax == 0:
                ol_smiles = df[df[prop] < th]['smiles'].values

            all_ol_smiles += ol_smiles.tolist()

            olp = 100*len(all_ol_smiles)/total

            df = df[~df.smiles.isin(ol_smiles)]
            df.reset_index(drop=True, inplace=True)


            if olp >= olp_th:
                break

        return df, all_ol_smiles




    


# def CV_all_pyods(df, rem_feats, scale=True, save_path="./"):
#     cv_scores, nol_dfs, ol_dfs = {}, {}, {}
#     for i, (clf_name, clf) in enumerate(classifiers.items()):
#         print(i, clf_name)
#         ol_pyod, nol_pyod = pyod_ols(clf, df, rem_feats, scale)
#         mrmse = get_crossval_score(nol_pyod)
#         cv_scores[clf_name] = mrmse
#         nol_dfs[clf_name] = nol_pyod
#         ol_dfs[clf_name] = ol_pyod
        
        
#         # ol_dfs.append(df=ol_pyod, rem_feats=['smiles', 'log_sol'])
#         nol_pyod.to_csv(f"./{save_path}/model_{i}.csv", index=False)

#     print(cv_scores)
#     return ol_dfs, nol_dfs





def get_ol_smiles(loc, df, prop_th_new):
    
    prop = prop_th_new.loc[loc, 'prop']
    th = prop_th_new.loc[loc, 'th']
    amax = prop_th_new.loc[loc, 'amax']
    
    if amax ==0:
        smiles = df[ df[prop] < th]['smiles'].values
    elif amax == 1:
        smiles = df[ df[prop] > th]['smiles'].values
        
    return smiles



# def pyod_ols(clf, df, rem_feats, scale=True):

#     sc = StandardScaler()

#     # using all the data
#     df2 = df.copy()
#     # drop features not used for fitting
#     X_df = df2.drop(rem_feats, axis=1)

#     # scale features if needed
#     if scale:
#         X_df = sc.fit_transform(X_df.values)
#     else:
#         X_df = X_df.values

#     # fit the outlier detector
#     clf.fit(X_df)
#     print(X_df.shape)

#     # make predictions for outliers
#     preds = clf.predict(X_df)

#     # outliers and non-outliers
#     # TODO: make sure that 0 corresponds
#     # to inliers and 1 to outliers for
#     # all the models.
#     nol_abod = df2.loc[preds==0, :]
#     ol_abod = df2.loc[preds==1, :]

#     return ol_abod, nol_abod

# from helper import get_crossval_score
# from config import classifiers

# def all_pyods(df, rem_feats, scale=True):
#     nol_dfs, ol_dfs = {}, {}
#     for i, (clf_name, clf) in enumerate(classifiers.items()):
#         print(i, clf_name)
#         df_tmp = df.copy()
#         print("dataset size: ", df_tmp.shape)
#         ol_pyod, nol_pyod = pyod_ols(clf, df_tmp, rem_feats, scale)
  
#         nol_dfs[clf_name] = nol_pyod
#         ol_dfs[clf_name] = ol_pyod

#     return ol_dfs, nol_dfs