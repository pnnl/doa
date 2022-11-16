import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from collections import Counter
# from kmeans_interp.kmeans_feature_imp import KMeansInterp
from sklearn.preprocessing import StandardScaler
# parameters for pyod
# from config import classifiers
#from utils import find_th
# , all_pyods
#from helper import get_crossval_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from PIL import Image, ImageDraw, ImageFilter
def s2m(s): return Chem.MolFromSmiles(s)
import random
import pickle
from PIL import ImageFont
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from PIL import Image, ImageDraw, ImageFilter
from rdkit.Chem import Draw
import os
import sys
import config

from PIL import Image
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageFont
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#font = ImageFont.truetype("/System/Library/Fonts/SFCompactDisplay.ttf", 20)

# def most_ol_of_feat(loc, df_ol, df_th):
#     # get smiles corresponding to each feature
#     # threshold. we want to find the most outlying
#     # smiles corresponding to each threshold.
#     s0 = get_ol_smiles(loc, df_ol, df_th)
#     # sort the smiles so that the largest outliers
#     # are on the top.
#     s0 = enpls[enpls.smiles.isin(s0)].sort_values(by="olness", ascending=False)
    
#     # find smiles that has an outlier value greater than the threshold
#     s0 = s0[s0.olness>olness_ol]
#     s0 = s0.smiles.values
#     # s0 = enpls[enpls.smiles.isin(s0)].sort_values(by="olness", ascending=False).iloc[:12,:]['smiles'].values
#     print(df_th.loc[loc, 'prop'], len(s0))
    
#     # return these smiles
#     return Chem.Draw.MolsToGridImage([s2m(i) for i in s0], molsPerRow=6, maxMols=12), s0

def run_kmeans(n_trials, X, df_ol_str):
    """
    run kmeans several times
    """
    tmp = pd.DataFrame()
    rand_states=[]
    for i in tqdm(range(n_trials)):
        print(f"working on kmeans group {i}")
        random_state = random.randrange(10000000)
        rand_states.append([i, random_state])
        
        model = KMeans(random_state=random_state)
        visualizer = KElbowVisualizer(model, k=(4,30))

        visualizer.fit(X)        # Fit the data to the visualizer
        n_clusters = visualizer.elbow_value_
        
        km = KMeans(n_clusters=n_clusters, random_state=random_state)
        km.fit(X)

        tmp.loc[:, f'c{i}'] =  km.labels_

    tmp.loc[:, 'smiles'] = df_ol_str.smiles
    
    return tmp, rand_states


def get_overlap_smiles_for_cluster(n_trials, cluster_num, tmp, smiles_in_same_cluster):
    # tmp is the dataframe containing cluster labels for each trial
    # smiles belonging to cluster_num in the first trial
    curr_c_smiles = tmp.loc[tmp.loc[:, 'c0']  == cluster_num, 'smiles'].values
    
    # go through rest of the trials and find for each trial,
    # smiles of which cluster are maximally overlapped with the
    # curr_c_smiles
    for trial in range(1,n_trials):
        max_overlap = -10 # set initial number of overlaps
        max_ol_cluster = -10 # set the overlapping cluster number
        
        unq_clusts = tmp.loc[:, f'c{trial}'].unique()
        
        # go through each cluster
        for ic in unq_clusts:

            # pick a particular cluster
            locs = (tmp.loc[:, f'c{trial}']  == ic).values
            # corresponding smiles
            other_c_smiles = tmp.loc[locs, 'smiles'].values
            
            # overlapping smiles
            overlap_smiles = list(set(other_c_smiles).intersection(curr_c_smiles))
            overlap = len(overlap_smiles)

            # if max found, set max_overlap to overlap
            # max overlap cluster to ic, the currernt cluster
            if overlap > max_overlap:
                max_overlap = overlap
                max_ol_cluster = ic

        max_overlap_smiles = tmp[tmp.loc[:, f'c{trial}'] == max_ol_cluster].smiles.values

        smiles_in_same_cluster[cluster_num].extend(max_overlap_smiles.tolist())
        

def find_overlaps(res_trials, n_trials):

    # uniq clusters for trial 0
    unq_clusts = res_trials.loc[:, 'c0'].unique()
    
    smiles_in_same_cluster = {i:[] for i in unq_clusts}
#     smiles_in_same_cluster = {i:[] for i in range(n_clusters)}
    
    
    
    for i in unq_clusts:
        curr_c_smiles = res_trials.loc[res_trials.loc[:, 'c0']  == i, 'smiles'].values
        smiles_in_same_cluster[i].extend(curr_c_smiles.tolist())

    for i in unq_clusts:
        get_overlap_smiles_for_cluster(n_trials, i, res_trials, smiles_in_same_cluster)
        
    return smiles_in_same_cluster

def get_final_clusters(smiles_in_same_cluster, th=10):
    
    final_cluster_smiles = {}

    for i in smiles_in_same_cluster.keys():

        c = Counter(smiles_in_same_cluster[i])
        # in how many trials did a smile appear
        c = pd.DataFrame(c.most_common(), columns=['smiles', 'ntimes'])

        if c[c.ntimes>=th].shape[0]>0:
            final_cluster_smiles[i] = c[c.ntimes>=th].smiles.values.tolist()
        
    return final_cluster_smiles


    

def save_cluster_imgs_1(final_cluster_smiles, cluster_keys, clstfol='clusters'):
    
    for cnum in cluster_keys:
        img = Draw.MolsToGridImage( [s2m(i) for i in final_cluster_smiles[cnum][:6] ],
                                        molsPerRow=6,returnPNG=False)
        img.save(f"./{clstfol}/grid_img_{cnum}.png")
    #     print(cnum)

def save_cluster_imgs_for_group(final_cluster_smiles, cluster_keys, clstfol='clusters'):
    
    for i in cluster_keys:

    #     c = df_ol_str.loc[kms.labels_== i, :]
        mols = [s2m(j) for j in final_cluster_smiles[i] ][:3]

        if len(mols)>0:
            [Draw.MolToFile(m, f'./{clstfol}/image_groups/grp{i}_mol{j}.png', size=(200, 200) ) for (j, m) in enumerate(mols)]
        else:
            print("the cluster has no mols")
        #     mols = [s2m(j) for j in c.smiles.values[:3]]
        #     [Draw.MolToFile(m, f'./{clstfol}/image_groups/grp{i}_mol{j}.png', size=(200, 200)) for (j, m) in enumerate(mols)]

def get_more_than_one_clusters(final_cluster_smiles):
    
    clusters_plot={}
    for k, v in final_cluster_smiles.items():
        if len(v) >1:
            clusters_plot[k]=v
            
    return clusters_plot


# use easily explainable features
def get_explainable_features():
    from mordred import Calculator, descriptors
    
    n_all = Calculator(descriptors, ignore_3D=False).descriptors
    n_2D = Calculator(descriptors, ignore_3D=True).descriptors

    modules=[]
    for i in n_2D:
        modules.append(i.__module__.split('.')[1])

    dict_modules = {k:[] for k in set(modules)}
    # modules=[]
    for i in n_2D:
        module = i.__module__.split('.')[1]
        n = str(i)
        dict_modules[module].extend([n])
        # break

    exaplainable = ['Aromatic', 'AtomCount', 'CPSA', 'TopoPSA', 'LogS', 'AcidBase',
                    'McGowanVolume', 'FragmentComplexity','CarbonTypes', 'BertzCT', 'BondCount',
                    'Polarizability', 'Weight', 'RotatableBond', 'RingCount', 'HydrogenBond']

    expl_props=[]
    for i in exaplainable:
        # print(dict_modules[i])
        expl_props.extend(dict_modules[i])
        
    return expl_props


# main kmeans function
def kmeans_main(clstfol='clusters_ol', df_input=None, rerun_clusters = True, n_trials = 10):
    # ol_or_nol = "ol"
    
    
#     if ol_or_nol == "ol":
#         # X = df_ol_str.drop(['log_sol', 'smiles'], axis=1)
#         clstfol = f"clusters_{n_trials}"
#     elif ol_or_nol == "nol":
#         # X = df_nol.drop(['log_sol', 'smiles'], axis=1)
#         clstfol = f"clusters_{n_trials}_nol"


    X = df_input.drop(['log_sol', 'smiles'], axis=1)
    X_cols = X.columns.tolist()
    sc = StandardScaler()
    X = sc.fit_transform(X) # features are standard scaled

    # df_ol_str.shape
    if rerun_clusters:
        
        print("removing old files ...")
        if os.path.exists(f"./{clstfol}"):
            os.system(f"rm -r ./{clstfol}")
     
        
        os.mkdir(f"./{clstfol}")
        os.mkdir(f"./{clstfol}/image_groups")
   
        print(f"running kmeans {n_trials} times...")
        res_trials, rand_states = run_kmeans(n_trials=n_trials, X=X, df_ol_str=df_input) # 1
    
        print("saving trial results...")
        res_trials.to_csv(f"./{clstfol}/res_trials.csv", index=False)

        print("saving rand_states...")
        with open(f"./{clstfol}/rand_states.pkl", "wb") as f:
            pickle.dump(rand_states, f)
            
            
        smiles_in_same_cluster = find_overlaps(res_trials=res_trials, n_trials = n_trials) # 2
        final_cluster_smiles = get_final_clusters(smiles_in_same_cluster, th = int(n_trials*.85)) #3
        with open(f'./{clstfol}/final_cluster_smiles.pkl', 'wb') as f:
            pickle.dump(final_cluster_smiles, f)
            
    else:
        print("loading clusters ...")
        with open(f'./{clstfol}/final_cluster_smiles.pkl', 'rb') as f:
            final_cluster_smiles = pickle.load(f)
            
    # final {cluster: smiles} dictionary
    n_final_clusters = len(final_cluster_smiles.keys())
    cluster_keys = np.sort(list(final_cluster_smiles.keys()))
    print(cluster_keys)

    # cluster_feature_weights = run_classification(final_cluster_smiles, df_input, cluster_keys)
    # save_cluster_imgs_1(final_cluster_smiles, cluster_keys)
    save_cluster_imgs_for_group(final_cluster_smiles, cluster_keys, clstfol = clstfol)
    
    clusters_plot = get_more_than_one_clusters(final_cluster_smiles)
    cluster_keys = np.sort( list(clusters_plot.keys()) )
    
#     if not os.path.exists(f"./{clstfol}"):
#         os.mkdir(f"./{clstfol}")
#         os.mkdir(f"./{clstfol}/image_groups")

#     if rerun_clusters:

#         print("saving res_trials...")
#         if not os.path.exists(f"./{clstfol}/res_trials.csv"):
#             res_trials.to_csv(f"./{clstfol}/res_trials.csv", index=False)

#         print("saving rand_states...")
#         if not os.path.exists(f"./{clstfol}/rand_states.pkl"):
#             with open(f"./{clstfol}/rand_states.pkl", "wb") as f:
#                 pickle.dump(rand_states, f)

#     if rerun_clusters:
#         print("running new cluster calculations ...")
#         os.system(f"rm ./{clstfol}/grid_img_*")
#         smiles_in_same_cluster = find_overlaps(res_trials=res_trials, n_trials = n_trials) # 2
#         final_cluster_smiles = get_final_clusters(smiles_in_same_cluster, th = int(n_trials*.85)) #3

#     if not os.path.exists(f'./{clstfol}/final_cluster_smiles.pkl'):
#         print("saving final cluster smiles...")
#         with open(f'./{clstfol}/final_cluster_smiles.pkl', 'wb') as f:
#             pickle.dump(final_cluster_smiles, f)


    return cluster_keys, clstfol, final_cluster_smiles



def remove_correlated(df):
    
    df_cp = df.copy()
    df_cp.drop(['log_sol', 'smiles'], axis=1, inplace=True)
    corr_matrix = df_cp.corr().abs()

    cols = corr_matrix.columns.values

    # cols.remove('avg_error')
    hc_pairs, hc_todrop = [], []
    for i in tqdm(range(len(cols))):
        for j in range(i+1, len(cols) ):
    #         j=0
            cv = corr_matrix[ cols[i] ][ cols[j] ]
    #         print(high_corr[i],  high_corr[j],  cv)
            if abs(cv) >.95:
                hc_pairs.append([cols[i], cols[j] ])
                hc_todrop.append(cols[j])

    # len(list(set(hc_todrop)))

    df_unc= df.drop(hc_todrop, axis=1)

    return df_unc



def grouped_Images(clstfol, cluster_keys, font, text_pos, n_rows=5, save_old_new=False, save_loc=None):
    
    files = os.listdir(f"./{clstfol}/image_groups/")
    img = Image.new('RGB', (1800,n_rows*200), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # for j in range(1,15):
    j1=0
    lbl=1
    old_new = {}
    # for jj in range(0, 21, 3):
    cl = 0

    if len(cluster_keys)%3 == 0:
        main_range = len(cluster_keys)
    else:
        main_range = len(cluster_keys)-3

    for jj in range(0, main_range, 3):
        print("row = ", jj)

        for jiter in range(3):

            j = cluster_keys[jj + jiter]
            nmols = len([f for f in files if f.startswith(f"grp{j}_mol")])


            for i in range(nmols):

                im1 = Image.open(f'./{clstfol}/image_groups/grp{j}_mol{i}.png')
                img.paste(im1,  (i*200 + jiter*600, j1*200))
            cl +=1
            old_new[j] = cl
            
            rel_loc, offset = text_pos[jj + jiter].split('.')
            offset = float(offset)
            #            row loc        col loc 
            if rel_loc == 'right':
                text_loc = (i*5 + jiter*600 + 450 + offset, j1*200+10)
            elif rel_loc == 'center':
                text_loc = (i*5 + jiter*600 + 225 + offset, j1*200+10)
            else:
                text_loc = (i*5 + jiter*600, j1*200+10)
            draw.text(text_loc, "Group "+str(cl) ,(0,0,0), font=font)


        j1+=1

    jj = jj+3
    for jiter in range(len(cluster_keys)%3):

        j = cluster_keys[jj + jiter]
        print("row d = ", j)
        nmols = len([f for f in files if f.startswith(f"grp{j}_mol")])


        for i in range(nmols):

            im1 = Image.open(f'./{clstfol}/image_groups/grp{j}_mol{i}.png')
            img.paste(im1,  (i*200 + jiter*600, j1*200))

        cl += 1
        old_new[j] = cl    
        draw.text((i*200 + jiter*600, j1*200), "Group "+str(cl) ,(0,0,0), font=font)


#     color = (52, 164, 235)
    color = (0,0,0)
    draw.line((600,0, 600, 1400), fill=color, width=4)
    draw.line((1200,0, 1200, 1400), fill=color, width=4)
    # draw.line((1800,0, 1800, 1000), fill=color, width=4)

    for i in range(1, n_rows+1):
        draw.line((0,i*200, 1800, i*200), fill=color, width=4)



    img.save(f"./{clstfol}/image_groups.png")
    if save_loc:
        img.save(f"./{save_loc}")
    if save_old_new:
        with open(f"{clstfol}/old_new.pkl", "wb") as f:
            pickle.dump(old_new,f)
        
    return img, old_new



def run_classification(all_data, final_cluster_smiles, cluster_keys, save_path):
    
    print('running new classification')
    km_data = all_data.copy()
    cluster_feature_weights = {}
    
    for group_label in cluster_keys:

        current_group = final_cluster_smiles[ group_label ]
        km_data.loc[:, 'new_label'] = 0 # reset the labels
        km_data.loc[km_data.smiles.isin(current_group), 'new_label'] = 1 # target group is assigned label = 1

        # seperate features and labels and standard scale
        X_new = km_data.drop(['smiles', 'log_sol', 'new_label'], axis=1)
        X_cols = X_new.columns.tolist()
        sc = StandardScaler()
        X_new.loc[:,X_cols] = sc.fit_transform(X_new.values)
        labels = km_data.new_label.values

        # train the classifier
        clf = ExtraTreesClassifier()
        clf.fit(X_new.values, labels)

        # get the feature importance values
        fimp = clf.feature_importances_
        # sort them in the descending order
        sorted_fimp = np.argsort(fimp)[::-1]

        # sorted feature importance labels
        fimp_ordered_features = np.array(X_cols)[sorted_fimp]
        
        # sorted feature importance values
        fimp_ordered_fvalues = fimp[sorted_fimp]

        cluster_feature_weights[group_label] = list(zip(fimp_ordered_features, 
                                                  fimp_ordered_fvalues))
        
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(cluster_feature_weights, f)
        
        
    return cluster_feature_weights

# def run_classification_old(save_path=None, final_cluster_smiles=None, ol_str=None, cluster_keys=None, save_weights=True):
    
#     ol_str = ol_str.copy()
#     ol_str['new_label'] = np.nan

#     for cl, sm in final_cluster_smiles.items():
#         ol_str.loc[ol_str.smiles.isin(sm), 'new_label'] = cl

#     ol_str.dropna(subset=['new_label'], inplace=True)


#     X_new = ol_str.drop(['smiles', 'log_sol', 'new_label'], axis=1)
#     X_cols = X_new.columns.tolist()
#     sc = StandardScaler()
#     X_new.loc[:,X_cols] = sc.fit_transform(X_new.values)

#     print("n_features: ", len(X_cols))
#     ordered_feature_names = X_cols


#     labels = ol_str.new_label.values
#     cluster_feature_weights = {}
#     for label in cluster_keys:
#         print(label)
#         binary_enc = np.vectorize(lambda x: 1 if x == label else 0)(labels)
#         clf = RandomForestClassifier()
#         clf.fit(X_new.values, binary_enc)

#     # Origianl code
#     #     sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
#     #     ordered_cluster_features = np.take_along_axis(
#     #         np.array(ordered_feature_names), 
#     #         sorted_feature_weight_idxes, 
#     #         axis=0)
#     #     ordered_cluster_feature_weights = np.take_along_axis(
#     #         np.array(clf.feature_importances_), 
#     #         sorted_feature_weight_idxes, 
#     #         axis=0)


#         fimp = clf.feature_importances_
#         sorted_feature_weight_idxes = np.argsort(fimp)[::-1]

#         features_ordered = np.array(ordered_feature_names)[sorted_feature_weight_idxes]
#         fimp_ordered = fimp[sorted_feature_weight_idxes]

#     #     cluster_feature_weights[label] = list(zip(ordered_cluster_features, 
#     #                                               ordered_cluster_feature_weights))

#         cluster_feature_weights[label] = list(zip(features_ordered, 
#                                                   fimp_ordered))
        
#     if save_weights:
# #             with open(f'{save_path}/cluster_feature_weights.pkl', 'wb') as f:
#         with open(save_path, 'wb') as f:
#             pickle.dump(cluster_feature_weights, f)
        
        
#     return cluster_feature_weights




# def plot_feature_imporatnce(cluster_feature_weights, cluster_keys, clstfol, bar_color, save_loc, nrows=4,
#                            minmax=False):
    
#     sns.set_style('white')
    
    
#     fig, ax = plt.subplots(nrows,3, figsize=(14,9), sharey=True )
#     axes = ax.ravel()

#     new_cluster_feature_weights = {i:None for i in cluster_keys}
#     for ic, cnum in enumerate(cluster_keys):

#         nf = 5
# #         a = np.array(cluster_feature_weights[cnum])[:nf]
        
#         a = np.array(cluster_feature_weights[cnum])
#         a[:,1] = a[:,1].astype(float)
        
#         if minmax:
#             mm = MinMaxScaler()
#             a[:,1] = mm.fit_transform(a[:,1].reshape(-1,1)).ravel()
#             new_cluster_feature_weights = a
#         a = a[:nf]
        

#         axes[ic].bar(np.arange(nf), a[:,1].astype(float), color=bar_color, width=.5)
#         axes[ic].set_xticks(np.arange(nf)) # <--- set the ticks first
#         # axes[ic].set_xticks(np.arange(nf)-1) # <--- set the ticks first
#         axes[ic].set_xticklabels(a[:,0], fontweight='bold', rotation=20)
#         # axes[ic].set_xticklabels(a[:,0], fontweight='bold', rotation=40)
#         axes[ic].set_title(f"cluster {ic+1}", fontweight='bold')
        
# #         if ic ==14:
# #             break

#     fig.text(0.07, 0.5, 'Feature Importance', va='center', rotation='vertical', fontweight='bold', fontsize=14)
#     plt.subplots_adjust(hspace=.8)
#     # plt.subplots_adjust(hspace=.8)
#     # plt.tight_layout()
#     plt.savefig(f"./{clstfol}/cluster_fimp.png")
#     plt.savefig(f"./{save_loc}")
#     return new_cluster_feature_weights

    
# def difference_plot(df_outs, df_ref, cluster_keys, cluster_feature_weights, final_cluster_smiles, save_name):
# #     save_name= 'images/unsup/ood_feature_difference.png'
    
#     fv_differences = {i:{} for i in cluster_keys }
    
#     sc = MinMaxScaler()
# #     sc = StandardScaler()
#     feats = df_ref.drop(['smiles', 'log_sol'], axis=1).columns.tolist()
#     df_ref_X = df_ref.copy()
#     df_ref_X.loc[:, feats] = sc.fit_transform(df_ref.loc[:, feats])

#     fig, ax = plt.subplots(len(cluster_keys)//3,3, figsize=(16,10))
#     axes = ax.ravel()

#     for j, clst_lbl  in enumerate(cluster_keys):

#         dfo =  df_ref_X[df_ref_X.smiles.isin(final_cluster_smiles[ clst_lbl ] )]
#         dfin = df_ref_X[~df_ref_X.smiles.isin(final_cluster_smiles[ clst_lbl ])]

#         fimp = [i[0] for i in cluster_feature_weights[ clst_lbl ][:5]]

#         fvs_ol = [dfo[i].mean() for i in fimp]
#         fvs_in = [dfin[i].mean() for i in fimp]

#         fv_differences[clst_lbl]['fvs_ol'] = fvs_ol
#         fv_differences[clst_lbl]['fvs_in'] = fvs_in


#         axes[j].bar(np.arange(5), fvs_ol, width =.2, color='b')
#         axes[j].bar(np.arange(5)+.1, fvs_in, width=.2, color='r')
#         axes[j].set_xlabel(f'Cluster {j+1}', fontsize=12, fontweight='bold')
#         axes[j].set_xticks(np.arange(5), fimp, fontsize=10, fontweight='bold', rotation=20)
        
#         if j == 0:
#             axes[j].legend(bbox_to_anchor=(.26, 1.5), loc="upper right", prop={'size':11, 'weight':'bold'})

#         # fig, ax = plt.subplots(4,3, figsize=(12,8))
#         # axes = ax.ravel()
#         # for i, key in enumerate(fv_differences.keys()):
#         #     axes[i].bar(np.arange(5), fv_differences[key]['fvs_ol'], width =.2, color='b')
#         #     axes[i].bar(np.arange(5)+.1, fv_differences[key]['fvs_in'], width=.2, color='r')
#     plt.subplots_adjust(hspace=1)
#     fig.text(0.08, 0.5, 'Descriptor Value', va='center', rotation='vertical', fontsize=12, fontweight='bold')
#     plt.savefig(save_name)
#     return fv_differences


class PlotImportance:
    def __init__(self, cluster_feature_weights=None, final_cluster_smiles=None,
                cluster_keys=None, clstfol=None, df_ref=None, nrows=4, ncols=3, minmax=True,
                 nf=5, diff_scaler='minmax'):
        
        self.cluster_feature_weights=cluster_feature_weights
        self.final_cluster_smiles=final_cluster_smiles
        self.cluster_keys= cluster_keys
        self.clstfol=clstfol
        self.nrows=nrows
        self.ncols = ncols
        self.minmax=minmax
        self.df_ref=df_ref
        self.nf = nf
        self.diff_scaler = diff_scaler


    def plot_feature_importance_and_difference(self, 
                                               diff_to_common_scale=True,
                                               imp_bar_color="#E74C3C",
                                               diff_bar_color="#E74D4C",
                                               save_img=None,
                                               scaled_cluster_feature_weights_path=None,
                                               difference_dfs_path=None,
                                               figsize=(14,9),
                                               hspace= .8,
                                               group_labels=None,
                                               bbox_to_anchor =(.26, 1.5),
                                               ytick_fs = 12,
                                               xtick_fs = 12,
                                               xtick_rot = 20,
                                               ylabel = "Feature Importance / Feature Difference",
                                               ylabel_pad=0.07,
                                               sharey=True,
                                               fimp_to_diff_max=True
                                              ):

        plt.rc('font', weight='bold')
#         plt.rc('ytick.major', size=5, pad=7)
        plt.rc('ytick', labelsize=ytick_fs)
        
        fig, ax = plt.subplots(self.nrows, self.ncols, figsize=figsize, sharey=sharey )
        axes = ax.ravel()
    
        scaled_cluster_feature_weights = {i:None for i in self.cluster_keys}
        difference_dfs = {i:None for i in self.cluster_keys}

        if self.diff_scaler == 'minmax':
            scaler = MinMaxScaler()
        elif self.diff_scaler == 'standard':
            scaler = StandardScaler()
            
        feats = self.df_ref.drop(['smiles', 'log_sol'], axis=1).columns.tolist()
        df_ref_X = self.df_ref.copy()
        df_ref_X.loc[:, feats] = scaler.fit_transform(self.df_ref.loc[:, feats])
        
        for ic, cnum in enumerate(self.cluster_keys):

            # scale feature importance
            a = np.array(self.cluster_feature_weights[cnum])
            a[:,1] = a[:,1].astype(float)

            if self.minmax:
                mm = MinMaxScaler()
                a[:,1] = mm.fit_transform(a[:,1].reshape(-1,1)).ravel()
                scaled_cluster_feature_weights[cnum] = a

            a = a[:self.nf]

            # outlier smiles
            dfo =  df_ref_X[df_ref_X.smiles.isin(self.final_cluster_smiles[ cnum ] )]
            # inlier smiles
            dfin = df_ref_X[~df_ref_X.smiles.isin(self.final_cluster_smiles[ cnum ])]

            # important feature labels 
            fimp = [i[0] for i in self.cluster_feature_weights[ cnum ]  ]

            # find the mean feature values
            fvs_ol = [dfo[i].mean() for i in fimp] # feature means of outlier smiles
            fvs_in = [dfin[i].mean() for i in fimp] # feature means of other smiles

            
            # find the feature value difference
            # min max scale the difference to get them to a common scale
            difference = (np.array(fvs_ol) - np.array(fvs_in)).reshape(-1,1)
#             if scale_differences:
#                 mm = StandardScaler()
#                 difference = mm.fit_transform( difference )
                
                
            # select first nf features. these are the most important features according to tree classifier
            
            # divide by the max difference in the chosen list. note that a difference
            # of 1 doesn't mean that this feature has the largest difference. only the largest 
            # among the chosen ones (first nf ).
            difference_for_plot = np.copy(difference)
            if diff_to_common_scale:
                difference_for_plot = difference_for_plot/np.max(np.abs(difference_for_plot))
                difference_for_plot = difference_for_plot[:self.nf]
            else:
                difference_for_plot = difference_for_plot[:self.nf]
                


            df_difference = pd.DataFrame(zip(fimp, difference.ravel()), columns=['feature', 'difference'])
            df_difference = df_difference.sort_values(by=['difference'], ascending=False)
            df_difference.loc[:, 'fvsign'] = 1
            df_difference.loc[df_difference.difference < 0, 'fvsign'] = -1

            difference_dfs[cnum] = df_difference

            if fimp_to_diff_max:
                scaled_fimp = a[:,1].astype(float) * np.max(abs(difference_for_plot.ravel()))
            else:
                scaled_fimp = a[:,1].astype(float)
                
            axes[ic].bar(np.arange(self.nf), scaled_fimp, color=imp_bar_color, width=.2, label='FI')
            axes[ic].bar(np.arange(self.nf)+.2, difference_for_plot.ravel(), color=diff_bar_color, width=.2, label='DIFF')
            axes[ic].set_xticks(np.arange(self.nf)) # <--- set the ticks first
            # axes[ic].set_xticks(np.arange(nf)-1) # <--- set the ticks first
            axes[ic].set_xticklabels(a[:,0], fontsize=xtick_fs, fontweight='bold', rotation=xtick_rot);
#             axes[ic].set_yticklabels(axes[ic].get_yticks(), weight='bold');
            # axes[ic].set_xticklabels(a[:,0], fontweight='bold', rotation=40)
            if group_labels:
                axes[ic].set_title(f"{group_labels[cnum]}", fontweight='bold')
            else:
                axes[ic].set_title(f"Group {ic+1}", fontweight='bold')

            if ic == 0:
                axes[ic].legend(bbox_to_anchor=bbox_to_anchor, loc="upper right", prop={'size':11, 'weight':'bold'})


        fig.text(ylabel_pad, 0.5, ylabel, va='center', rotation='vertical', fontweight='bold', fontsize=14)
        plt.subplots_adjust(hspace=hspace)
        # plt.subplots_adjust(hspace=.8)
        # plt.tight_layout()
        # plt.savefig(f"./{clstfol}/cluster_fimp.png")
        if save_img:
            plt.savefig(f"./{save_img}")

        if scaled_cluster_feature_weights_path:
            with open(f'{self.clstfol}/{scaled_cluster_feature_weights_path}', 'wb') as f:
                pickle.dump(scaled_cluster_feature_weights, f)
        if difference_dfs_path:
            with open(f'{self.clstfol}/{difference_dfs_path}', 'wb') as f:
                pickle.dump(difference_dfs, f)

        # return new_cluster_feature_weights
        return scaled_cluster_feature_weights, difference_dfs
                  
                  
                  
    def plot_feature_difference(self, difference_dfs, diff_bar_color="#E74C3C", save_path=None):
                  
        fig, ax = plt.subplots(self.nrows, 3, figsize=(14,9), sharey=True )
        axes = ax.ravel()

        for ic, cnum in enumerate(self.cluster_keys):

            data = difference_dfs[cnum]
            data.reset_index(drop=True, inplace=True)
            fvalues = data.loc[:self.nf-1, 'difference'].values.astype(float)
            fvalues = fvalues/np.max(fvalues)
            fnames = data.loc[:self.nf-1, 'feature'].values

            axes[ic].bar(np.arange(self.nf), fvalues, color=diff_bar_color, width=.2, label='DIFF')
        #     axes[ic].bar(np.arange(nf)+.2, difference_for_plot.ravel(), color=diff_bar_color, width=.2, label='DIFF')
            axes[ic].set_xticks(np.arange(self.nf)) # <--- set the ticks first
            # axes[ic].set_xticks(np.arange(nf)-1) # <--- set the ticks first
            axes[ic].set_xticklabels(fnames, fontweight='bold', rotation=20)
            # axes[ic].set_xticklabels(a[:,0], fontweight='bold', rotation=40)
            axes[ic].set_title(f"Group {ic+1}", fontweight='bold')

            if ic == 0:
                axes[ic].legend(bbox_to_anchor=bbox_to_anchor, loc="upper right", prop={'size':11, 'weight':'bold'})

            plt.subplots_adjust(hspace=.8)
                  
        fig.text(0.07, 0.5, 'Feature Difference', va='center', rotation='vertical', fontweight='bold', fontsize=14)
        plt.savefig(f"./{save_path}")