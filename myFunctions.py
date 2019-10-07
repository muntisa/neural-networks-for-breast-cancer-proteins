import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score,f1_score, recall_score, precision_score
from sklearn.utils import class_weight

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif

import seaborn as sns; sns.set() # data visualization library 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from imblearn.over_sampling import SMOTE

import joblib

def DataPreprocessing(df):
    # remove duplicates
    # remove NA values
    # Check the number of data points in the data set
    nNAs=0
    
    print('\n-> Dataset preprocessing')
    print('Inicial shape:', df.shape)
    
    print("Data points =", len(df))
    
    # Check the number of columns in the data set
    print("Columns (output + features)=",len(df.columns))
    
    # Check the data types
    print("Data types =", df.dtypes.unique())
    
    # Dataset statistics
    print('\n')
    df.describe()
    
    # print names of columns
    print('Column Names:\n', df.columns)
    
    # see if there are categorical data
    print("Categorical features:", df.select_dtypes(include=['O']).columns.tolist())
    
    # Check NA values
    # Check any number of columns with NaN
    print("Columns with NaN: ", df.isnull().any().sum(), ' / ', len(df.columns))

    # Check any number of data points with NaN
    nNAs = df.isnull().any(axis=1).sum()
    print("No of data points with NaN:", nNAs, ' / ', len(df))
    
    # remove duplicates
    print('* Remove duplicates')
    df.drop_duplicates(keep=False, inplace=True)
    
    # remove columns with NA values
    if nNAs !=0:
        print('* Remove columns with NA values')
        for col in df.columns:
            df = df[df[col].notnull()]
        
    print('Final shape:', df.shape)
    print('Done!')
    return df

def ClearDatasets(df):
    # change last column with Class, remove few columns, keep ProtID!
    print('\n-> Modify dataset')
    df.columns = list(df.columns[:-2])+["ProtID","Class"]
    # drop few columns (keep only descriptors + Class)
    df= df.drop(['Unnamed: 0', 'V3'],axis = 1)
    print('Done!')
    return df

def  set_weights(y_data, option='balanced'):
    """Estimate class weights for umbalanced dataset
       If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y)). 
       If a dictionary is given, keys are classes and values are corresponding class weights. 
       If None is given, the class weights will be uniform """
    cw = class_weight.compute_class_weight(option, np.unique(y_data), y_data)
    w = {i:j for i,j in zip(np.unique(y_data), cw)}
    return w


def DataCheckings(df):
    # CHECKINGS ***************************
    # Check the number of data points in the data set
    nNAs=0
    
    print('\n-> Checking dataset')
    
    print("\nData points =", len(df))
    
    # Check the number of columns in the data set
    print("\nColumns (output + features)=",len(df.columns))
    
    # Check the data types
    print("\nData types =", df.dtypes.unique())
    
    # Dataset statistics
    print('\n')
    df.describe()
    
    # print names of columns
    print('Column Names:\n', df.columns)
    
    # see if there are categorical data
    print("\nCategorical features:", df.select_dtypes(include=['O']).columns.tolist())
    
    # Check NA values
    # Check any number of columns with NaN
    print("\nColumns with NaN: ", df.isnull().any().sum(), ' / ', len(df.columns))

    # Check any number of data points with NaN
    nNAs = df.isnull().any(axis=1).sum()
    print("\nNo of data points with NaN:", nNAs, ' / ', len(df))
    
    print('Done!')
    return nNAs
    

def getDataFromDataFrame(df, OutVar='Class'):
    # get X, Y data and column names from df
    print('\n-> Get X & Y data, Features list')
    print('Shape', df.shape)
    
    # select X and Y
    ds_y = df[OutVar]
    ds_X = df.drop(OutVar,axis = 1)
    Xdata = ds_X.values # get values of features
    Ydata = ds_y.values # get output values

    print('Shape X data:', Xdata.shape)
    print('Shape Y data:', Ydata.shape)
    
    # return data for X and Y, feature names as list
    print('Done!')
    return (Xdata, Ydata, list(ds_X.columns))


def Remove0VarCols(df):
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class 
    print('\n-> Remove zero variance features')
    # print('Initial features:', Features)
    selector= VarianceThreshold()
    Xdata = selector.fit_transform(Xdata)
    # Selected features
    SelFeatures = []
    for i in selector.get_support(indices=True):
        SelFeatures.append(Features[i])
    print('Removed features:',list(set(Features) - set(SelFeatures)))
    
    # create the resulted dataframe
    df = pd.DataFrame(Xdata,columns=SelFeatures)
    df['Class'] = Ydata # add class column
    # print('Final columns:', list(df.columns))
    print('Done!')
    return df


def ScaleDataFrame(df,sFile):
    # scale dataframe, save scaled file, save scaler
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class 

    # Standardize dataset
    scaler = MinMaxScaler()
    Xdata = scaler.fit_transform(Xdata)

    df = pd.DataFrame(Xdata,columns=Features)
    df['Class'] = Ydata # add class column

    scalerFile = sFile[:-4]+'.scaler_MinMax.pkl'
    print('* Save scaler:', scalerFile)
    joblib.dump(scaler, scalerFile) 

    # Save initial ds
    scaledFile = sFile[:-4]+'.ds_m.csv'
    print('* Save scaled dataset:', scaledFile)
    df.to_csv(scaledFile, index=False)
    print('Done!')
    return df

def FeatureSelection(df,sFile,nFeats=1):
    if nFeats == 0:
        print("\n NO feature selection!")
        return df
    # Feature selection
   
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class
    print('\n-> Univariate Feature selection')
    # print('Initial columns:', list(df.columns))
    selector= SelectKBest(chi2, k=nFeats)
    Xdata = selector.fit_transform(Xdata, Ydata)
    
    selectorFile = sFile[:-4]+'.featSelector_Univariate'+str(nFeats)+'.pkl'
    print('* Save selector:', selectorFile)
    joblib.dump(selector, selectorFile) 
    
    # Selected features
    SelFeatures = []
    for i in selector.get_support(indices=True):
        SelFeatures.append(Features[i])
        
    # create the resulted dataframe
    df = pd.DataFrame(Xdata,columns=SelFeatures)
    df['Class'] = Ydata # add class column
    print('Final columns:', list(df.columns))
    
    # Save selected feature ds
    selectFile = sFile[:-4]+'.ds_sel.csv'
    print('* Save selected features dataset:', selectFile)
    df.to_csv(selectFile, index=False)
    
    print('Done!')
    return df


def PCAtransform(df,sFile,nPCA=2):
    
    if nPCA == 0:
        print("\n NO PCA reduction!")
        return df
    
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class
   
    print('\n-> PCA dimension reduction')
    # print('Initial columns:', list(df.columns))
    pcaModel = PCA(n_components=nPCA)
    Xdata = pcaModel.fit_transform(Xdata)

    pcaFile = sFile[:-4]+'.PCA'+str(nPCA)+'.pkl'
    print('* Save PCA model:', pcaFile)
    joblib.dump(pcaModel, pcaFile) 

    # create the resulted dataframe
    PCAfeatures =[]
    for p in range(nPCA):
        PCAfeatures.append('PCA'+str(p+1))
    PCAfeatures

    df = pd.DataFrame(Xdata,columns=PCAfeatures)
    df['Class'] = Ydata # add class column
    print('Final columns:', list(df.columns))

    # Save selected feature ds
    sFile_PCA = sFile[:-4]+'.ds_PCA.csv'
    print('* Save PCA dataset:', sFile_PCA)
    df.to_csv(sFile_PCA, index=False)

    print('Done!')
    return df


def SMOTEdf(df,sFile,seed=0):
    # SMOTE balanced df
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class
    print('\n-> Dataframe SMOTE balancing')
    print('Initial dimensions:', df.shape)
    
    smote = SMOTE(ratio='minority',random_state=seed)
    X_sm, y_sm = smote.fit_sample(Xdata, Ydata)
    
    # create the resulted dataframe
    df = pd.DataFrame(X_sm,columns=Features)
    df['Class'] = y_sm # add class column
    print('Final shape:', df.shape)
    
    # Save SMOTE balanced ds
    smoteFile = sFile[:-4]+'.ds_bal.csv'
    print('* Save balanced dataset:', smoteFile)
    
    df.to_csv(smoteFile, index=False)
    print('Done!')
    return df

def MLOuterCV(Xdata, Ydata, label = 'my', class_weights = {0: 1, 1: 1}, folds = 3, seed = 74):
    # inputs:
    # data for X, Y; a label about data, number of folds, seeed
    
    # default: 3-fold CV, 1:1 class weights (ballanced dataset)
    
    # define classifiers
    names = ['NB','KNN','LDA','SVM linear','SVM','LR','MLP','DT','RF','XGB','GB','AdaB','Bagging'] # ,
    
    priors = [(class_weights[0]/(class_weights[0]+class_weights[1])), (class_weights[1]/(class_weights[0]+class_weights[1]))]
    
    classifiers = [GaussianNB(),
                   KNeighborsClassifier(3),
                   LinearDiscriminantAnalysis(solver='svd',priors=priors), # No tiene random_state
                   SVC(kernel="linear",random_state=seed,gamma='scale',class_weight=class_weights),
                   SVC(kernel = 'rbf', random_state=seed,gamma='scale',class_weight=class_weights),
                   LogisticRegression(solver='lbfgs',random_state=seed,class_weight=class_weights),
                   MLPClassifier(hidden_layer_sizes= (20), random_state = seed, max_iter=50000, shuffle=False),
                   DecisionTreeClassifier(random_state = seed,class_weight=class_weights),
                   RandomForestClassifier(n_jobs=-1,random_state=seed,class_weight=class_weights),
                   XGBClassifier(n_jobs=-1,seed=seed,scale_pos_weight= class_weights[0]/class_weights[1]),
                   GradientBoostingClassifier(random_state=seed),
                   AdaBoostClassifier(random_state = seed),
                   BaggingClassifier(random_state=seed)
                  ]
    # results dataframe: each column for a classifier
    df_res = pd.DataFrame(columns=names)

    # build each classifier
    print('* Building scaling+feature selection+outer '+str(folds)+'-fold CV for '+str(len(names))+' classifiers:', str(names))
    total = time.time()
    
    # define a fold-CV for all the classifier
    outer_cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    
    print('ML method, Mean, SD, Time (min)')
    for name, clf in zip(names, classifiers):
        start = time.time()
        
        # clf.fit(Xdata,Ydata)
        
        # evaluate pipeline
        scores = cross_val_score(clf, Xdata, Ydata, cv=outer_cv, scoring='roc_auc', n_jobs=-1)
        
        df_res[name] = scores
        print('%s, %0.3f, %0.4f, %0.1f' % (name, scores.mean(), scores.std(), (time.time() - start)/60))
        
    print('Total time:', (time.time() - total)/60, ' mins')             
    
    # return AUC scores for all classifiers as dataframe (each column a classifier)
    return df_res


def MyML(sFile, summaryFile, boxplotFile, nSel=0, nPCA=0, outVar='Class',nfold=3, seed=74):
    # nSel = 0 -> no Feature selection
    # df correction, preprocessing, scale, feature selection, PCA, SMOTE, ML
    
    # READ dataset with original descriptors
    print('\n-> Read dataset', sFile)
    df = pd.read_csv(sFile)
    # print(list(df.columns))

    # Clean columns (remove all extra columns but keep ProtID!)
    df = ClearDatasets(df)
    # print(list(df.columns))

    # drop ProtID to have only descriptors + Class (raw dataset)
    print('\n-> Drop ProtID column')
    df= df.drop(['ProtID'],axis = 1)
    print('Done!')
    # print(list(df.columns))

    # Check dataset
    # nNAs = DataCheckings(df)

    # Dataset preprocessing
    df = DataPreprocessing(df)
    # print(list(df.columns))

    # Remove zero variance columns
    df = Remove0VarCols(df)

    # Save initial ds
    rawFile = sFile[:-4]+'.ds_raw.csv'
    print('\n-> Save raw dataset:',rawFile)
    df.to_csv(rawFile, index=False)
    print('Done!')

    # Scale raw dataframe
    df = ScaleDataFrame(df,sFile)

    # Univariate feature selection
    df = FeatureSelection(df,sFile,nFeats=nSel)
    
    # PCA dimension reduction
    df = PCAtransform(df,sFile,nPCA=nPCA)

    # Balancing dataframe using SMOTE
    df = SMOTEdf(df,sFile,seed)

    # get ds for ML
    Xdata, Ydata, Features = getDataFromDataFrame(df)# out var = Class 

    # ML n-fold CV (outer)
    class_weights = set_weights(Ydata)
    print("Class weights = ", class_weights)

    df_results = None # all results 
    df_fold = MLOuterCV(Xdata, Ydata, label = outVar, class_weights = class_weights, folds = nfold, seed = seed)
    df_fold['Dataset'] = outVar
    df_fold['folds'] = nfold

    # add each result to a summary dataframe
    df_results = pd.concat([df_results,df_fold])

    # save all results
    
    print('\n==>> Saving summary', summaryFile)
    df_results.to_csv(summaryFile, index=False)

    # save boxplot
    classifierNames = list(df_results.columns)
    classifierNames.remove('Dataset')
    classifierNames.remove('folds')

    foldTypes=[nfold]

    plt.figure()
    plt.clf()
    print('==> Fold =', nfold)
    grouped = df_results[df_results['folds']==nfold].drop(['folds'], axis=1).groupby('Dataset')
    grouped.boxplot(figsize=(16,12), return_type='axes')
    plt.title("")
    plt.xlabel("Machine Learning methods",size=18)
    plt.ylabel("Mean AUROC (3-fold CV)",size=18)
    plt.tick_params(labelsize=14)
    plt.ylim(0.5,1.0)
    plt.savefig(boxplotFile, dpi=1200)
    plt.savefig(boxplotFile)
    plt.show()

    df_results
    return
