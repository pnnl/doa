import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import RandomizedSearchCV

from hyperopt import Trials, STATUS_OK, tpe
# from hyperas import optim
# from hyperas.distributions import choice, uniform

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# %matplotlib inline

from scipy.stats import spearmanr
from keras.initializers import random_normal, random_uniform
# from keras.initializers import normal

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from rdkit import Chem

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import pickle
import json
import pandas as pd
#from utils import get_transformed_data, create_model
# from keras.models import load_model
from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
import os
# from keras.callbacks import ModelCheckpoint
import random
from tqdm import tqdm
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler



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



def run_ol(part, calcs, df, features):
    
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
    with open(f'smiles_error_sup_et_{part}.pkl', 'wb') as f:
        pickle.dump(smiles_error, f)
