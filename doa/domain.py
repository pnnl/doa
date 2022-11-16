import pandas as pd
import mordred
from mordred import Calculator, descriptors
from rdkit import Chem
import openbabel
import os
from tqdm import tqdm
import pickle
from openbabel import pybel
from scipy.spatial import ConvexHull
import numpy as np
from scipy.stats import moment
import config

def get_domain_data(doa, data, dom_id):
    out_domain, in_domain = [], []
    for i in range(dom_id+1):
#         print('domain ', i)
        
        prop = doa.loc[i, 'prop']
        direction = doa.loc[i, 'amax']
        direction2 = doa.loc[i, 'direction']
        th = doa.loc[i, 'th']

        # out of domain molecules have property values greater than the threshold
        if direction == 1:
            out_domain.extend(data[data[prop] > th].smiles.values.tolist())
            
        # out of domain molecules have property values less than the threshold
        elif direction == 0:
            out_domain.extend(data[data[prop] < th].smiles.values.tolist())
                    
    out_domain = list(set(out_domain))
    in_domain = data[~data.smiles.isin(out_domain)].smiles.values.tolist()
    
    return out_domain, in_domain


def get_dom_percent(dom, ol_list):
    ovrlap = []
    for d in ol_list:
        n = len(list(set(dom).intersection(d)))
        p = n/len(dom)
        ovrlap.append( p )
    return ovrlap

def get_domain_percents(doa, all_data, ol_list):
    
    sum_res = []
    for idom in tqdm(doa.index):

        # get ind and ood molecules for a given domain
        outd, ind = get_domain_data(doa, all_data, idom)


        # how many ood in (data outlier, structural outlier, structural anomaly, and inlier
        # how many ind in (data outlier, structural outlier, structural anomaly, and inlier



        # percentage of data_ols, str_ols, str_anomaly, inliers in out of domain set
        ovrlap_ood = get_dom_percent(outd, ol_list)
        # percentage of data_ols, str_ols, str_anomaly, inliers in out of in-domain set
        ovrlap_ind = get_dom_percent(ind, ol_list)

        sum_res.append(ovrlap_ood + ovrlap_ind)

        
    sum_res = pd.DataFrame(sum_res, columns=['data_ols_in_ood', 'str_ols_in_ood', 'str_anm_in_ood', 'inl_in_ood',
                                             'data_ols_in_ind', 'str_ols_in_ind', 'str_anm_in_ind', 'inl_in_ind']
                          )

    return sum_res


def train_cls(df_il, df_ol):
    
    df_il = df_il.copy()
    df_ol = df_ol.copy()
#     df2 = df_str_ols.copy()

    df_il.loc[:,'label'] = 1
    df_ol.loc[:,'label'] = 0

    dff = pd.concat([df_il, df_ol], axis=0)
    dff.reset_index(drop=True, inplace=True)

    train, test = train_test_split(dff, test_size=.2)

    train_x = train.drop(['smiles', 'log_sol', 'label'], axis=1)
    test_x = test.drop(['smiles', 'log_sol', 'label'], axis=1)

    train_y = train['label'].values
    test_y = test['label'].values



    et = ExtraTreesClassifier()

    et.fit(train_x.values, train_y)

    pred = et.predict(test_x.values)

    print(sum(pred == test_y)/len(pred))
    
    cls_dict = classification_report(y_pred=pred, y_true=test_y, output_dict=True)
    
    return cls_dict






def get_all_data():
    
    train = pd.read_csv("/people/pana982/solubility/data/full_dataset/sets/train.csv")
    val = pd.read_csv("/people/pana982/solubility/data/full_dataset/sets/val.csv")
    all_data = pd.concat([train, val], axis=0)
    all_data.reset_index(drop=True, inplace=True)
    
    pnnl = all_data.shape[0]

    cui = pd.read_csv("/people/pana982/solubility/data/cui_data/mdm_cui.csv")
    cui = cui.shape[0]
    
    delaney = pd.read_csv("/people/pana982/solubility/data/delaney/mdm_delaney.csv")
    delaney = delaney.shape[0]
    
    return pnnl, cui, delaney


def get_ols_for_datasets(doa, pubchem_mdm):
    
    df_ref = pubchem_mdm.copy()
    ss = doa.copy()

    cv_results = []
    all_outsmiles=[]
    ol_test_smiles = []
    nol_test = []
    saved = 0
    for i in doa.index:
        prop = ss.loc[i, 'prop']
        direction = ss.loc[i, 'amax']
        th = ss.loc[i, 'th']
        
        if prop not in df_ref.columns:
            print('prop not in pubchem data.. have to think about a fix')
            continue


    #     df_train = train.copy()
    #     df_test = test.copy()

    #     print(prop, th)

        if direction == 1:
            all_outsmiles.extend(df_ref[df_ref[prop] > th].smiles.values.tolist())
    #         ol_test_smiles.extend(df_test[df_test[prop] > th].smiles.values.tolist())

        elif direction == 0:
            all_outsmiles.extend(df_ref[df_ref[prop] < th].smiles.values.tolist())
    #         ol_test_smiles.extend(df_test[df_test[prop] < th].smiles.values.tolist())
    
    
    all_outsmiles = list(set(all_outsmiles))
    in_smiles = pubchem_mdm[~pubchem_mdm.smiles.isin(all_outsmiles)]
    
    return all_outsmiles, in_smiles



def inout_domain_by_olpcnt(pnnl_fbf, cui_fbf, del_fbf,
                           pubchem_mdm,
                           cutoff_percents=[0.05, 0.1, 0.15, 0.20]):
    

    res=[]
    for cutoff_percent in cutoff_percents:

        """
            we check at which domain (which row in the doa table) the total number of
            outliers gets greater than a given cutoff percentage. for each dataset, we
            determine this. this is the doa we consider to find how many molecules are 
            inside this domain and how many are outside.
        """
        pnnl_cutoff = pnnl_fbf[pnnl_fbf.all_olp > cutoff_percent ]
        cui_cutoff = cui_fbf[cui_fbf.all_olp > cutoff_percent ]
        del_cutoff = del_fbf[del_fbf.all_olp > cutoff_percent ]

        pnnl_cutoff_id = pnnl_cutoff.index[0]
        cui_cutoff_id = cui_cutoff.index[0]
        del_cutoff_id = del_cutoff.index[0]

        pnnl_out_pcnt = pnnl_cutoff.all_olp.values[0]
        cui_out_pcnt = cui_cutoff.all_olp.values[0]
        del_out_pcnt = del_cutoff.all_olp.values[0]

#         pnnl_out_pcnt = pnnl_fbf[pnnl_fbf.all_olp > cutoff_percent].all_ol_percent.values[0]
#         cui_out_pcnt = cui_fbf[cui_fbf.all_olp > cutoff_percent].all_ol_percent.values[0]
#         del_out_pcnt = del_fbf[del_fbf.all_olp > cutoff_percent].all_ol_percent.values[0]


        # selecting the domains for each dataset based on the cutoff index
        pnnl_sub = pnnl_fbf.loc[:pnnl_cutoff_id , :]
        cui_sub = cui_fbf.loc[:cui_cutoff_id, :]
        delaney_sub = del_fbf.loc[:del_cutoff_id, :]

        
        # finding the number of in-domain and out-of-domain
        # molecules in the pubchem set based on the doa determined
        # using each dataset.
        # PNNL
        out_pnnl, in_pnnl = get_ols_for_datasets(pnnl_sub, pubchem_mdm)
        n_out_pnnl = len(set(out_pnnl))
        n_in_pnnl = in_pnnl.shape[0]


        # CUI
        out_cui, in_cui = get_ols_for_datasets(cui_sub, pubchem_mdm)
        n_out_cui = len(set(out_cui))
        n_in_cui = in_cui.shape[0]


        # DELANEY
        out_delaney, in_delaney = get_ols_for_datasets(delaney_sub, pubchem_mdm)
        n_out_delaney = len(set(out_delaney))
        n_in_delaney = in_delaney.shape[0]


        res.append([n_out_pnnl, n_in_pnnl, pnnl_out_pcnt,
                    n_out_cui, n_in_cui, cui_out_pcnt,
                    n_out_delaney, n_in_delaney, del_out_pcnt])



    res = pd.DataFrame(res, columns=['out_pnnl', 'in_pnnl', 'pnnl_out_pcnt',
                    'out_cui', 'in_cui', 'cui_out_pcnt',
                    'out_delaney', 'in_delaney', 'del_out_pcnt'])
    
    return res


def in_out_domain_vs_r2(doa, pubchem_mdm):
    
    """
    how many molecules are in each domain of applicability.
    a single domain corresponds to a single row in doa threshold table.
    here, we find how many molecules are inside the domain for each domain 
    of applicability.
    
    r2 is the r2 achieved on an arbitrary test set after removing out-of-domain
    molecules corresponding to a particular domain from that test set.
    out-of-domain molecules are not removed from the train set.
    """
    df_ref = pubchem_mdm.copy()
    ss = doa.copy()

    all_outsmiles=[]
    res=[]
    for i in range(20):
        prop = ss.loc[i, 'prop']
        direction = ss.loc[i, 'direction']
        th = ss.loc[i, 'th']
        r2 = ss.loc[i, 'r2']
        all_out_pcnt = ss.loc[i, 'all_olp']


    #     df_train = train.copy()
    #     df_test = test.copy()
    #     print(prop, th)

        if direction == 1:
            all_outsmiles.extend(df_ref[df_ref[prop] > th].smiles.values.tolist())
    #         ol_test_smiles.extend(df_test[df_test[prop] > th].smiles.values.tolist())

        elif direction == 0:
            all_outsmiles.extend(df_ref[df_ref[prop] < th].smiles.values.tolist())
    #         ol_test_smiles.extend(df_test[df_test[prop] < th].smiles.values.tolist())
    
        all_outsmiles = list(set(all_outsmiles))
        in_smiles = pubchem_mdm[~pubchem_mdm.smiles.isin(all_outsmiles)]
        
        res.append([len(all_outsmiles), in_smiles.shape[0], r2,  all_out_pcnt])
    
    res = pd.DataFrame(res, columns=['out_domain', 'in_domain', 'r2', 'olp'])
    
    return res