import warnings
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

class classifier_choose():
    
    def __init__(self,target,df,class_weight=None):

        self.df = df
        self.target = target

        self.pca_mark = 'off'
        self.standard_mark = 'off'
        self.minmax_mark = 'off'
        
        self.X = self.df.drop({self.target},axis=1)
        self.y = self.df[self.target]

        self.models_dic_base = {'RandomForestClassifier':RandomForestClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier(),
        'LogisticRegression':LogisticRegression(),'XGBClassifier':XGBClassifier(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier()}

        self.model_dict_for_grid = {

            RandomForestClassifier() : {'n_estimators': [64, 100, 128],'class_weight': [class_weight]},
            GradientBoostingClassifier() : {'n_estimators': [64, 100, 128]},
            LogisticRegression() : {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(np.log10(0.01), np.log10(100.0), num=4),
            'solver': ['lbfgs', 'sag', 'saga'],'class_weight': [class_weight]},
            XGBClassifier() :{'max_depth': [3, 5, 7],'eta': np.logspace(np.log10(0.1), np.log10(3.0), num=3),
            'subsample': np.linspace(0.1, 0.5, 3),'min_split_loss': [ 0.1, 0.2, 0.3],
            'alpha': [ 0.01, 0.1, 1],'lambda': [ 0.01, 0.1, 1],'min_child_weight': [1, 5, 10],
            'booster': ['gbtree', 'gblinear']},
            SVC() : {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},
            KNeighborsClassifier() : {'n_neighbors' : [1,2,4,8,16,32], 'weights' : ['uniform', 'distance']}}


        self.model_dict_for_random = {

            RandomForestClassifier() : 
            {'n_estimators': range(64, 128, 1),'class_weight': [class_weight]},

            GradientBoostingClassifier() : 
            {'n_estimators': range(64, 128, 1)},

            LogisticRegression() : 
            {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(np.log10(0.01), np.log10(100.0), num=10),
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'class_weight': [class_weight],'fit_intercept' : [True,False], 'max_iter' : [100,200,500]},

            XGBClassifier() :
            {'max_depth': [3, 4, 5, 6, 7],'eta': np.logspace(np.log10(0.1), np.log10(4.0), num=10),
            'subsample': np.linspace(0.1, 0.5, 5),'min_split_loss': [0, 0.1, 0.2, 0.3, 0.4],
            'alpha': [0, 0.001, 0.01, 0.1, 1],'lambda': [0, 0.001, 0.01, 0.1, 1],
            'min_child_weight': [1, 3, 5, 7, 10],'booster': ['gbtree', 'gblinear']},

            SVC() : 
            {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},

            KNeighborsClassifier() : 
            {'n_neighbors' : range(1, 30, 1), 'weights' : ['uniform', 'distance']}}

    def split_data(self,valid = 0.15,test=0.15,stratify=None):

        if stratify:
            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test, stratify=self.y)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid,stratify=y)

        else:
            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid)

        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test

    def preprocessing(self,mode='StandardScaler',n_components=2):

        if mode=='MinMaxScaler':

            self.standard_mark = 'on'
            scaler = MinMaxScaler()

        if mode=='StandardScaler':

            self.standard_mark = 'on'
            scaler = StandardScaler()

        if mode=='MinMaxScaler' or mode=='StandardScaler':

            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.scaler = scaler
                
        if mode=='PCA':

            self.pca_mark = 'on'

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            pca = PCA(n_components=n_components)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_valid = pca.transform(self.X_valid)
            self.X_test = pca.transform(self.X_test)

            self.X_train = pd.DataFrame(self.X_train)
            self.X_valid = pd.DataFrame(self.X_valid)
            self.X_test = pd.DataFrame(self.X_test)

            self.scaler = scaler
            self.pca = pca

        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test

    def preanalize(self,alpha=0.5):

        # correlation plot
        corr_df = pd.DataFrame(self.df).corr()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index, 
                    y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title(f"Feature Correlation to {self.target}")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        # distribution plot
        if len(pd.DataFrame(self.df)[f'{self.target}'].value_counts()) < 10:
            cluster_counts = pd.DataFrame(self.df)[f'{self.target}'].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.histplot(data=pd.DataFrame(self.df), x=f'{self.target}', kde=True, color='green', bins=20, ax=ax2)

        ax2.set_title(f"{self.target} Distribution")
        ax2.set_xlabel(f"{self.target}")
        ax2.set_ylabel("Count")

        plt.show()

        # PCA plot
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_train)

        X_train_pca = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(12,6))
        sns.scatterplot(x=X_train_pca[0],y=X_train_pca[1],data=pd.DataFrame(self.X),hue=self.y,alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def ensemble(self,tuner='RandomizedSearchCV',cv=5,scoring='accuracy',n_iter=5,n_jobs=None):

        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.result_df_ensamble = pd.DataFrame()
        self.ensemble_models = {}

        if tuner == "GridSearchCV":
            model_dict_for_tuner = self.model_dict_for_grid
        if tuner == "RandomizedSearchCV" or tuner == 'default':
            model_dict_for_tuner = self.model_dict_for_random

        for key, value in tqdm(model_dict_for_tuner.items(), desc="Tuning Ensemble Models"):
            ensemble_model_name = key.__class__.__name__
            print(f"Model: {ensemble_model_name}")

            if tuner == "GridSearchCV":
                ensemble_search = GridSearchCV(key,cv=cv,n_jobs=n_jobs,param_grid=value,scoring=scoring)
            if tuner == "RandomizedSearchCV":
                ensemble_search = RandomizedSearchCV(key,cv=cv,n_jobs=n_jobs,param_distributions=value,scoring=scoring,n_iter=n_iter)
            if tuner == 'default':
                ensemble_search = key

            ensemble_search.fit(self.X_train,self.y_train)
            self.ensemble_models[ensemble_model_name] = ensemble_search

            df_iter_model_ensemble = self.result_test_df(ensemble_search)
            df_iter_model_ensemble = df_iter_model_ensemble.transpose()
            df_iter_model_ensemble.rename(columns={df_iter_model_ensemble.columns[-1]: str(key)}, inplace=True)
            self.result_df_ensamble = pd.concat([self.result_df_ensamble, df_iter_model_ensemble], axis=1)
        self.result_df_ensamble = self.result_df_ensamble.transpose()
        self.result_df_ensamble.index = list(self.models_dic_base.keys())

        return self.result_df_ensamble.iloc[:, :6]

    def cv_results(self,estimator_from='ensemble',estimator=None,result='df',param=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel_model

        if estimator_from == 'ada':
            model = self.ada_model

        results = pd.DataFrame(model.cv_results_)
        parameter_names = first_key = list(results['params'][0].keys())
        parameter_names = ['param_' + param for param in parameter_names]
        parameter_names.append('mean_test_score')
        parameter_names.append('std_test_score')
        results.sort_values(by='mean_test_score', ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)

        if result == 'df':

            return results[parameter_names]

        if result == 'plot':

            results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)
            plt.ylabel('Mean test score')
            plt.xlabel('Hyperparameter combinations')

        if result == 'summarize_by':

            tmp = pd.concat([
                results.groupby('param_'+param)['mean_test_score'].mean(),
                results.groupby('param_'+param)['mean_test_score'].std()], axis=1)
            tmp.columns = ['mean_test_score', 'std_test_score']
            return tmp

    def voting(self,mode='hard'):

        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        try:
            self.voting_model = VotingClassifier(estimators=[(key,value.best_estimator_) for key,value in self.ensemble_models.items()],voting = mode)

        except AttributeError:
            self.voting_model = VotingClassifier(estimators=[(key,value) for key,value in self.ensemble_models.items()],voting = mode)

        self.voting_model.fit(self.X_train,self.y_train)
        return self.result_test_df(self.voting_model)

    def result_test_df(self,model):

        y_pred_valid = model.predict(self.X_valid)
        accuracy_valid = round(accuracy_score(self.y_valid, y_pred_valid),2)
        precision_valid = round(precision_score(self.y_valid, y_pred_valid, average='macro'),2)
        recall_valid = round(recall_score(self.y_valid, y_pred_valid, average='macro'),2)
        f1_valid = round(f1_score(self.y_valid, y_pred_valid, average='macro'),2)

        y_pred_test = model.predict(self.X_test)
        accuracy_test = round(accuracy_score(self.y_test, y_pred_test),2)
        precision_test = round(precision_score(self.y_test, y_pred_test, average='macro'),2)
        recall_test = round(recall_score(self.y_test, y_pred_test, average='macro'),2)
        f1_test = round(f1_score(self.y_test, y_pred_test, average='macro'),2)

        try:
            result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[model.best_estimator_.get_params(),model.cv_results_['mean_fit_time'].sum(),
                accuracy_valid,precision_valid,recall_valid,f1_valid,accuracy_test,precision_test,recall_test,f1_test]},
                index=['parameters','building_time','accuracy_valid','precision_valid','recall_valid','f1_valid','accuracy_test','precision_test','recall_test','f1_test'])
        except AttributeError:
            result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[model.get_params(),'|',accuracy_valid,precision_valid,
                recall_valid,f1_valid,accuracy_test,precision_test,recall_test,f1_test]},
                index=['parameters','building_time','accuracy_valid','precision_valid','recall_valid','f1_valid','accuracy_test','precision_test','recall_test','f1_test'])

        result_test_df = result_test_df.transpose()
        
        return result_test_df

    def ada(self,n_estimators=50,learning_rate=1.0,estimator_from='ensemble',estimator='LogisticRegression'):

        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if estimator_from == 'empty':
            weak_estimator = self.models_dic_base[estimator]

        if estimator_from == 'ensemble':
            weak_estimator = self.ensemble_models[estimator].best_estimator_

        if estimator_from == 'voting':
            weak_estimator = self.voting_model

        if estimator_from == 'basemodel':
            weak_estimator = self.basemodel_model.best_estimator_

        try:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        except:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME.R', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        return self.result_test_df(self.ada_model)

    def basemodel(self,mode='auto_random',estimator='LogisticRegression',params=None,cv=5,scoring='accuracy',n_iter=10,n_jobs=None):

        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        model = self.models_dic_base[estimator]

        index_for_params_random = {key.__class__.__name__: index for index, key in enumerate(list(self.model_dict_for_random.keys()))}
        parameter_dic_random = self.model_dict_for_random[list(self.model_dict_for_random.keys())[index_for_params_random[estimator]]]

        index_for_params_grid = {key.__class__.__name__: index for index, key in enumerate(list(self.model_dict_for_grid.keys()))}
        parameter_dic_grid = self.model_dict_for_grid[list(self.model_dict_for_grid.keys())[index_for_params_grid[estimator]]]

        if mode == 'set_manual':
            search = model.set_params(**params)

        if mode == 'set_grid':
            search = GridSearchCV(model,cv=cv,n_jobs=n_jobs,param_grid={**params},scoring=scoring)

        if mode == 'auto_grid':
            search = GridSearchCV(model,cv=cv,n_jobs=n_jobs,param_grid=parameter_dic_grid,scoring=scoring)

        if mode == 'set_random':
            search = RandomizedSearchCV(model,cv=cv,n_jobs=n_jobs,param_distributions={**params},scoring=scoring,n_iter=n_iter)

        if mode == 'auto_random':
            search = RandomizedSearchCV(model,cv=cv,n_jobs=n_jobs,param_distributions=parameter_dic_random,scoring=scoring,n_iter=n_iter)

        search.fit(self.X_train, self.y_train)
        self.basemodel_model = search

        return self.result_test_df(search).iloc[:, :6]
        
    def get_pipe(self,estimator_from,estimator=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel_model

        if estimator_from == 'ada':
            model = self.ada_model

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.scaler,model)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.scaler,self.pca,model)

        return self.build_pipe

    def plot_mat(self,estimator_from='voting',estimator=None):   

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel_model

        if estimator_from == 'ada':
            model = self.ada_model

        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = model.classes_
        ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
        ax.set(xlabel='Predict', ylabel='Actual')

        print(classification_report(self.y_test,y_pred))
 
    def pca_heat(self,n=100,vs=18,sh=4,dpi=150):

        df_comp = pd.DataFrame(self.pca.components_,columns=self.df.drop({self.target},axis=1).columns)

        plt.figure(figsize=(vs,sh),dpi=dpi)
        sns.heatmap(df_comp[:n],annot=True)

    def pca_choose(self,min_n=2,max_n=10):
            
        scaler = StandardScaler()
        pca_X = scaler.fit_transform(self.X)

        explained_variance = []

        for n in range(min_n,max_n):
            pca = PCA(n_components=n)
            pca.fit(pca_X)
            
            explained_variance.append(np.sum(pca.explained_variance_ratio_))

        plt.plot(range(min_n,max_n),explained_variance)
        plt.xlabel("Number of Components")
        plt.ylabel("Variance Explained")
        plt.grid(alpha=0.2);
