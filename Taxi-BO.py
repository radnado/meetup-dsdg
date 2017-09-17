# Bayesian Optomization for Hyperparameter Tuning
# Coded for Windows 10, PTVS interative interpreter 
# Uses data from https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta
import datetime as dt

import pickle as pkl

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
import seaborn as sns
sns.set() # set as default plot style

from bayes_opt import BayesianOptimization 
import xgboost as xgb
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')

#region ### Setup 
import os
import sys
os.chdir(os.path.dirname(sys.argv[0]))
os.getcwd()

import time
start_time = time.time()

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.width', 0)

dataFolder = 'Data/'
foldFolder = 'Fold/'
loadFolder = 'Load/'
submitFolder = 'Submit/'
#endregion

ID = 'id'
TARGET = 'trip_duration'
np.random.seed(4242)


#region ###### Backup of xgb.cv version ##############
#def xgbFunction(max_depth, subsample, min_child_weight, gamma, colsample_bytree):
#def xgbFunction(max_depth, min_child_weight):
#    # Evaluate an XGBoost model using given params
#    xgb_params = {
#        'n_trees': 150,
#        'eta': 0.5,
#        'max_depth': int(max_depth),
#        'subsample': 0.7,   #max(min(subsample, 1), 0),
#        'objective': 'reg:linear',

#        'nthread': 10,  
#        'seed': 2017,

#        # GPU parameters
#        #'updater': 'grow_colmaker,prune', # CPU (default)
#        'updater': 'grow_gpu',             # GPU 'grow_gpu_hist'
#        #'tree_method': 'gpu_exact', # Standard Xgboost
#        #'tree_method': 'gpu_hist',   # Accellerated Xgboost - more mem, not as accurate
#        #'predictor':'gpu_predictor', # or 'predictor':'cpu_predictor'.
        
#        'eval_metric': 'rmse', 
#        'silent': 1,
#        'min_child_weight': int(min_child_weight),
#        'gamma': 0,  # max(gamma, 0),
#        'lambda': 0,  # 1.0
#        'colsample_bytree': 0.7   # max(min(colsample_bytree, 1), 0)
#    }
#    scores = xgb.cv(xgb_params, dtrain, num_boost_round=60, 
#                    early_stopping_rounds=50, verbose_eval=False, 
#                    maximize=False, nfold=6)['test-rmse-mean'].iloc[-1]

#    #return -scores  # convert to negative as we are minimizing 
#    return 1/(scores**3)  # recip enables it to maximize 
#endregion ##############

#def xgbFunction(max_depth, min_child_weight, colsample_bytree, subsample):
def xgbFunction(xgbLambda):
    
    xgbParam = {
        'objective': 'reg:linear',
        'booster' : 'gbtree',
        'eval_metric': 'rmse', 
        'silent': 1,
        'n_jobs': -1,  
        'seed': 4242,

        # GPU parameters
        #'updater': 'grow_colmaker,prune', # CPU (default)
        #'updater': 'grow_gpu',            # GPU 'grow_gpu_hist'
        #'tree_method': 'gpu_exact',  # Standard Xgboost
        #'tree_method': 'gpu_hist',   # Accellerated Xgboost - more mem, not as accurate
        #'predictor':'gpu_predictor', # or 'predictor':'cpu_predictor'.

        ### Parameter Set 1
        #'max_depth': int(round(max_depth)),
        #'min_child_weight': int(round(min_child_weight)),
        #'colsample_bytree': max(min(round(colsample_bytree,3), 1), 0),    
        #'subsample': max(min(round(subsample,3), 1), 0),  

        #'lambda': 0, 
        #'gamma': 0,   
        #'eta': 0.5,
        #'n_trees': 100    
        
        ### Parameter Set 2
        'max_depth': 10,
        'min_child_weight': 87,
        'colsample_bytree': .4,  
        'subsample': 1.0,  

        'lambda': max(round(xgbLambda,3), 0), 
        
        'gamma': 0,   
        'eta': 0.3,
        'n_trees': 400     
        }

    cvScoreTotal = 0
    cvScoreMean = 0
        
    for f, fold in enumerate(foldList):
        # print('x>>', fold)   # 0 mTrain_   1 yTrain_   2 mEval_   3 yEval_
        mFold = eval(fold[0])
        yFold = eval(fold[1])
        mVal = eval(fold[2])
        yVal = eval(fold[3])
        # print(str(fold[0]) + ' num_row= ' + str(mFold.num_row()))
        
        watchlist = [(mFold, 'train'), (mVal, 'val')]
        xgbModel = xgb.train(xgbParam, mFold,
                             num_boost_round=400,
                             early_stopping_rounds=30,
                             evals=watchlist,
                             verbose_eval=False, 
                             maximize=False)  

        #print('Fold ' + str(f + 1) + ' best_score= ' + str(xgbModel.best_score) +
        #      ' best_iteration= ' + str(xgbModel.best_iteration))
        cvScoreTotal += xgbModel.best_score

        #del xgbModel  # remind gc to release memory for gpu

    # Calc cvScoreMedian and if best then save
    cvScoreMean = round(cvScoreTotal / len(foldList), 6)
    #print('    cvMean= ',cvScoreMean) 

    #return -scores  # convert to negative as we are minimizing 
    return 1/(cvScoreMean**3)  # recip enables it to maximize 


############################
if __name__ == '__main__':

    ##### Load Fold and Val
    print('\nLoading folds and val...')
    foldsToProcess = 3  # 1 to 10 
    foldList = []
    featureList = pkl.load(open("Data/Bel-feature_names.pkl", "rb"))
    for foldCount in range(1, foldsToProcess + 1):   
        print("Loading fold %s" % foldCount)
 
        exec("dFold" + str(foldCount) + "= pkl.load(open(r'Fold\dFold" + str(foldCount) + ".pkl', 'rb'))")

        #exec("yFold" + str(foldCount) + "= dFold" + str(foldCount) + "[TARGET]")
        exec("yFold" + str(foldCount) + "= np.log(dFold" + str(foldCount) + "[TARGET].values + 1)")

        exec("mFold" + str(foldCount) + "= xgb.DMatrix(sparse.csr_matrix(dFold" + str(foldCount) +
                                        "[featureList]), yFold" +  str(foldCount) +")")
    
        exec("dVal" + str(foldCount) + "= pkl.load(open(r'Fold\dVal" + str(foldCount) + ".pkl', 'rb'))")

        #exec("yVal" + str(foldCount) + "= dVal" + str(foldCount) + "[TARGET]")
        exec("yVal" + str(foldCount) + "= np.log(dVal" + str(foldCount) + "[TARGET].values + 1)")

        exec("mVal" + str(foldCount) + "= xgb.DMatrix(sparse.csr_matrix(dVal" + str(foldCount) +
                                        "[featureList]), yVal" +  str(foldCount) +")")

        exec("foldList.append(['mFold" + str(foldCount) + "','yFold" + str(foldCount) + "'," + 
                              "'mVal"  + str(foldCount) + "','yVal"  + str(foldCount) + "'])")
    #print(foldList) 


    #####################
    # BO - Xgboost parameters to search
    # Parameter Set 1
    #params = {
    #          'max_depth': (2, 18),           # 10   # (2, 16),  
    #          'min_child_weight':(20, 200),   # 50   # (1, 120),
    #          'colsample_bytree':(0.2, 1.0),  # 0.3  # (0.1, 1), 
    #          'subsample': (0.2, 1.0),        # 0.8  # (0.5, 1),
    #         }

    # Parameter Set 2
    params = {
              'xgbLambda': (0, 5)             # 0    # (0, 10) 
             }

    # Initialize BO optimizer object
    bayesOpt = BayesianOptimization(xgbFunction, params)

    ''' Maximize parameters:  
    # .maximize(init_points=2, n_iter=25, acq="ucb", kappa=1, **gp_params)
    #                             Exploit      Explore     
    # Upper Confidence Bound
    #   acq="ucb", kappa=?          1           10   
    #
    # Expected Improvement
    #   acq="ei", xi=?             0.0001      0.1  
    #
    # Probability of Improvement
    #   acq="poi", xi=?            0.0001      0.1 
    '''
    
    # Optional - add known parameter combinations and actual scores  
    # In future use Grid search to get these and initialise BO manually) 
    #bayesOpt.initialize({   'target': [-1, -1],
    #                        'max_depth':         [10, 12],          
    #                        'min_child_weight': [50, 100],   
    #                        'colsample_bytree': [0.5, 0.7],  
    #                        'subsample':        [0.5, 0.7]})
    
    # Initialise BO points
    bayesOpt.maximize(init_points=4, n_iter=0, acq="ei", xi=0.1)
    

    # Run BO with 'early stoping' check on each batch of 10 iterations
    batchSize  = 2        # iterations per batch 
    batchTotal = 100       # batchTotal x batchSize = maximum total iterations
    batchesEarlyStop = 10  # stop if no improvement in previous n batches 
    
    batchCount = 0
    batchesSinceImprove = 0 
    maxScorePrev = 0 
    xiValue = 0.1
    
    while (batchCount < batchTotal) & (batchesSinceImprove < batchesEarlyStop): 
        batchCount += 1
        if (batchCount == 10): xiValue = 0.05  # after 30 iterations increase bias to exploitation 

        bayesOpt.maximize(n_iter=batchSize, acq="ei", xi=xiValue)
        
        # convert parameters calculated with integers back to integers in the BO object
        for row in bayesOpt.res['all']['params']:
            for k, v in row.items():
                if (k == 'max_depth') | (k == 'min_child_weight'): 
                    row[k] = round(row[k])
                else: 
                    row[k] = round(row[k],3)

        for k, v in bayesOpt.res['max']['max_params'].items():
            if (k == 'max_depth') | (k == 'min_child_weight'): 
                bayesOpt.res['max']['max_params'][k] = round(bayesOpt.res['max']['max_params'][k])
            else: 
                bayesOpt.res['max']['max_params'][k] = round(bayesOpt.res['max']['max_params'][k],3)


        if (bayesOpt.res['max']['max_val'] > maxScorePrev):
            batchesSinceImprove = 0  # this batch improved score so reset to 0 
            maxScorePrev = bayesOpt.res['max']['max_val']
        else: 
            batchesSinceImprove += 1 

        # Interim results: Report top scores, save and prepare data for plots
        dBValues = pd.DataFrame(bayesOpt.res['all']['values'],columns=['values']) 
        dBParams = pd.DataFrame.from_dict(bayesOpt.res['all']['params']) 
        dBResults = pd.concat([dBValues, dBParams],axis=1)
        dBResults.sort_values('values',ascending=False,inplace=True)
        dBResults.to_csv('ModelXgb/dBResults.csv', index=False) # save to disk
        print('Top Scores after Batch', batchCount, ' (',batchCount*batchSize, ' iterations)')
        print(dBResults.head(6))
        print()

        # Save BO object to disk 
        pkl.dump(bayesOpt, open("ModelXgb/xgbBO.pkl", "wb"))
        #  bayesOpt = pkl.load(open("ModelXgb/xgbBO.pkl", "rb"))
    

    bestScore = bayesOpt.res['max']['max_val']
    print('Best Score=  ', bestScore, ' >>> ',    #1 / (16.858005023685497 ** (1./3.))
          np.round(1/ (bestScore  ** (1./3.)),6))  
    print('Best Params= ', bayesOpt.res['max']['max_params']) 
    
    
    #### Joint Plots 
    # Parameter Set 1
    #sns.jointplot(data=dBResults, x='max_depth', y='values', kind='reg', color='g')
    #sns.plt.draw()

    #sns.jointplot(data=dBResults, x='min_child_weight', y='values', kind='reg', color='g')
    #sns.plt.draw()

    #sns.jointplot(data=dBResults, x='colsample_bytree', y='values', kind='reg', color='g')
    #sns.plt.draw()

    #sns.jointplot(data=dBResults, x='subsample', y='values', kind='reg', color='g')
    #sns.plt.show()
 
    # Parameter Set 2
    sns.jointplot(data=dBResults, x='xgbLambda', y='values', kind='reg', color='g')
    sns.plt.show()  
    
    ##### Heatmap
    #bins = [0,2,4,6,8,10,12,14,16,18,20,22] # *** change to list comprehension
    #labels  = [0,2,4,6,8,10,12,14,16,18,20]
    #dBResults['max_depth_group'] = pd.cut(dBResults['max_depth'], bins, 
    #                                      right=False, labels=labels)

    #bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]
    #labels  = [0,10,20,30,40,50,60,70,80,90,100,110,120]
    #dBResults['min_child_weight_group'] = pd.cut(dBResults['min_child_weight'], bins, 
    #                                      right=False, labels=labels)

    #dHeat = dBResults.pivot_table(index='max_depth_group', columns='min_child_weight_group',
    #                        values='values', aggfunc=np.mean)
    #ax = sns.heatmap(dHeat, annot=True, fmt=".1f", vmin=14, vmax=17)
    #ax.invert_yaxis()
    #sns.plt.show()


    # Get the best params for passing on to the training with all train data
    #p = bayesOpt.res['max']['max_params']

    print('<<< END >>> {} minutes \n'.format(round((time.time() - start_time)/60, 1)))  