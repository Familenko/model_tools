import warnings
from tqdm import tqdm
import timeit

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    
    def __init__(self,target,df,class_weight=None,probability=True):

        self.df = df
        self.target = target

        self.pca_mark = 'off'
        self.standard_mark = 'off'
        self.minmax_mark = 'off'
        
        self.X = self.df.drop({self.target},axis=1)
        self.y = self.df[self.target]

        self.models_dic_base = {'DecisionTreeClassifier':DecisionTreeClassifier(),'RandomForestClassifier':RandomForestClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier(),
        'XGBClassifier':XGBClassifier(),'SVC':SVC(probability=probability),'LogisticRegression':LogisticRegression(),'KNeighborsClassifier':KNeighborsClassifier()}

        self.models_dic_grid = {

            DecisionTreeClassifier() : {'splitter' : ["best", "random"], 'max_features' : ["auto", "sqrt", "log2"]},
            RandomForestClassifier() : {'n_estimators': [64, 100, 128],'class_weight': [class_weight]},
            GradientBoostingClassifier() : {'n_estimators': [64, 100, 128]},
            XGBClassifier() :{'max_depth': [3, 5, 7],'eta': np.logspace(np.log10(0.1), np.log10(3.0), num=3),
            'subsample': np.linspace(0.1, 0.5, 3),'min_split_loss': [ 0.1, 0.2, 0.3],
            'alpha': [ 0.01, 0.1, 1],'lambda': [ 0.01, 0.1, 1],'min_child_weight': [1, 5, 10],
            'booster': ['gbtree', 'gblinear']},
            SVC(probability=probability) : {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},
            LogisticRegression() : {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(np.log10(0.01), np.log10(100.0), num=4),
            'solver': ['lbfgs', 'sag', 'saga'],'class_weight': [class_weight]},
            KNeighborsClassifier() : {'n_neighbors' : [1,2,4,8,16,32], 'weights' : ['uniform', 'distance']}}


        self.models_dic_random = {

            DecisionTreeClassifier() : 
            {'splitter' : ["best", "random"], 'max_features' : ["auto", "sqrt", "log2"], 'max_depth' : range(1,20)},

            RandomForestClassifier() : 
            {'n_estimators': range(64, 128, 1),'class_weight': [class_weight],'max_depth':range(1, 10, 1)},

            GradientBoostingClassifier() : 
            {'n_estimators': range(64, 128, 8)},

            XGBClassifier() :
            {'max_depth': [3, 4, 5, 6, 7],'eta': np.logspace(np.log10(0.1), np.log10(4.0), num=10),
            'subsample': np.linspace(0.1, 0.5, 5),'min_split_loss': [0, 0.1, 0.2, 0.3, 0.4],
            'alpha': [0, 0.001, 0.01, 0.1, 1],'lambda': [0, 0.001, 0.01, 0.1, 1],
            'min_child_weight': [1, 3, 5, 7, 10],'booster': ['gbtree', 'gblinear']},

            SVC(probability=probability) : 
            {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},

            LogisticRegression() : 
            {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(np.log10(0.01), np.log10(100.0), num=10),
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'class_weight': [class_weight],'fit_intercept' : [True,False], 'max_iter' : [100,200,500]},

            KNeighborsClassifier() : 
            {'n_neighbors' : range(1, 30, 1), 'weights' : ['uniform', 'distance']}}

    def split_data(self,valid = 0.15,test=0.15,stratify=None):

        start_time = timeit.default_timer()

        if stratify:
            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test, stratify=self.y)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid,stratify=y)

        else:
            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid)

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test

    def preprocessing(self,mode='StandardScaler',n_components=2):

        start_time = timeit.default_timer()

        if mode=='MinMaxScaler':

            self.minmax_mark = 'on'
            self.standard_mark = 'off'
            self.pca_mark = 'off'
            scaler = MinMaxScaler()

        if mode=='StandardScaler':

            self.minmax_mark = 'off'
            self.standard_mark = 'on'
            self.pca_mark = 'off'
            scaler = StandardScaler()

        if mode=='MinMaxScaler' or mode=='StandardScaler':

            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.scaler = scaler
                
        if mode=='PCA':

            self.minmax_mark = 'off'
            self.standard_mark = 'off'
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

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test

    def preanalize(self,alpha=0.5,bins=20):

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
            sns.histplot(data=pd.DataFrame(self.df), x=f'{self.target}', kde=True, color='green', bins=bins, ax=ax2)

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

    def ensemble(self,tuner='RandomizedSearchCV',cv=5,scoring='accuracy',n_iter=5,n_jobs=1,tree_only=False):

        start_time = timeit.default_timer()
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.result_df_ensamble = pd.DataFrame()
        self.ensemble_models = {}

        if tuner == "GridSearchCV":
            model_dict_for_tuner = self.models_dic_grid
        if tuner == "RandomizedSearchCV" or tuner == 'default':
            model_dict_for_tuner = self.models_dic_random

        for idx, (key, value) in enumerate(tqdm(model_dict_for_tuner.items(), desc="Tuning Ensemble Models")):

            if tree_only and idx >= 4:
                break

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

        if tree_only:
            self.result_df_ensamble.index = list(self.models_dic_base.keys())[:4]
        else:
            self.result_df_ensamble.index = list(self.models_dic_base.keys())

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.result_df_ensamble.iloc[:, :6]

    def voting(self,voting='hard'):

        start_time = timeit.default_timer()
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.voting_model = VotingClassifier(estimators=[(key, value.best_estimator_) if hasattr(value, 'best_estimator_')
        else (key, value) for key, value in self.ensemble_models.items()],voting=voting)

        self.voting_model.fit(self.X_train,self.y_train)

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.result_test_df(self.voting_model)

    def basemodel(self,mode='auto_random',estimator='LogisticRegression',params=None,cv=5,scoring='accuracy',n_iter=10,n_jobs=None):

        start_time = timeit.default_timer()
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        model = self.models_dic_base[estimator]

        index_for_params_random = {key.__class__.__name__: index for index, key in enumerate(list(self.models_dic_random.keys()))}
        parameter_dic_random = self.models_dic_random[list(self.models_dic_random.keys())[index_for_params_random[estimator]]]

        index_for_params_grid = {key.__class__.__name__: index for index, key in enumerate(list(self.models_dic_grid.keys()))}
        parameter_dic_grid = self.models_dic_grid[list(self.models_dic_grid.keys())[index_for_params_grid[estimator]]]

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

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.result_test_df(search).iloc[:, :6]

    def tuning(self,estimator_from='default',estimator='KNeighborsClassifier', target='n_neighbors',set_target=None,params=None, min_n=1, max_n=30,step=1,metric='accuracy'):

        start_time = timeit.default_timer()

        test_error_rates = []

        if estimator_from == 'default':
            tuning_model = self.models_dic_base[estimator]

        if estimator_from == 'ensemble':
            tuning_model = self.ensemble_models[estimator].best_estimator_
            params = self.ensemble_models[estimator].best_params_

        if estimator_from == 'basemodel':
            tuning_model = self.basemodel_model.best_estimator_
            params = self.basemodel_model.best_params_

        if estimator_from == 'ada':
            tuning_model = self.ada_model
            params = self.ada_model.get_params()

        if estimator_from == 'bagging':
            tuning_model = self.bagging_model
            params = self.bagging_model.get_params()

        if params and target in params:
            del params[target]

        if set_target:
            params_dict = {target : set_target}
            if params:
                params_dict.update(params)
            tuning_model.set_params(**params_dict)
            tuning_model.fit(self.X_train, self.y_train)

            self.tuning_model = tuning_model

            return self.result_test_df(tuning_model).iloc[:, :6]

        else:
            for i in tqdm(np.arange(min_n, max_n, step),desc="Checking model"):

                try:
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    tuning_model.set_params(**params_dict)

                except InvalidParameterError:
                    i = i.astype('int')
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    tuning_model.set_params(**params_dict)

                tuning_model.fit(self.X_train, self.y_train) 
                y_pred_test = tuning_model.predict(self.X_valid)

                if metric == 'accuracy':
                    metric_score = accuracy_score(self.y_valid, y_pred_test)

                if metric == 'precision':
                    metric_score = precision_score(self.y_valid, y_pred_test)

                if metric == 'recall':
                    metric_score = recall_score(self.y_valid, y_pred_test)

                if metric == 'f1':
                    metric_score = f1_score(self.y_valid, y_pred_test)

                test_error_rates.append(metric_score)

            plt.plot(np.arange(min_n, max_n, step), test_error_rates, label='Test Error')
            plt.legend()
            plt.ylabel(f'{metric}')
            plt.xlabel(f'{target}')
            plt.show()

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))

    def ada(self,n_estimators=50,learning_rate=1.0,estimator_from='default',estimator='DecisionTreeClassifier'):

        start_time = timeit.default_timer()
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if estimator_from == 'default':
            weak_estimator = self.models_dic_base[estimator]

        if estimator_from == 'ensemble':
            weak_estimator = self.ensemble_models[estimator].best_estimator_

        if estimator_from == 'voting':
            weak_estimator = self.voting_model

        if estimator_from == 'basemodel':
            weak_estimator = self.basemodel_model.best_estimator_

        if estimator_from == 'tuning':
            weak_estimator = self.tuning_model

        if estimator_from == 'bagging':
            weak_estimator = self.bagging_model

        try:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        except:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME.R', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        return self.result_test_df(self.ada_model)

    def bagging(self,estimator_from='default',estimator='DecisionTreeClassifier',n_estimators=500,max_samples=0.1,bootstrap=True,n_jobs=1,oob_score=True,max_features=1.0,bootstrap_features=True):

        start_time = timeit.default_timer()
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if estimator_from == 'default':
            weak_estimator = self.models_dic_base[estimator]

        if estimator_from == 'ensemble':
            weak_estimator = self.ensemble_models[estimator].best_estimator_

        if estimator_from == 'voting':
            weak_estimator = self.voting_model

        if estimator_from == 'basemodel':
            weak_estimator = self.basemodel_model.best_estimator_

        if estimator_from == 'tuning':
            weak_estimator = self.tuning_model

        if estimator_from == 'ada':
            weak_estimator = self.ada_model

        self.bagging_model = BaggingClassifier(weak_estimator,n_estimators=n_estimators,max_samples=max_samples,
            bootstrap=bootstrap,n_jobs=n_jobs,oob_score=oob_score,max_features=max_features,bootstrap_features=bootstrap_features)
        self.bagging_model.fit(self.X_train,self.y_train)

        print("Elapsed time:", round(timeit.default_timer() - start_time,2))
        print(f'oob_score - {self.bagging_model.oob_score_}')
        return self.result_test_df(self.bagging_model)

    def cv_results(self,estimator_from='ensemble',estimator=None,result='df',param=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'basemodel':
            model = self.basemodel_model

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
        
    def get_pipe(self,estimator_from,estimator=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel_model

        if estimator_from == 'ada':
            model = self.ada_model

        if estimator_from == 'bagging':
            model = self.bagging_model

        if estimator_from == 'tuning':
            model = self.tuning_model

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.scaler,model)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.scaler,self.pca,model)

        return self.build_pipe
 
    def pca_heat(self,n=100,vs=18,sh=4,dpi=150):

        df_comp = pd.DataFrame(self.pca.components_,columns=self.df.drop({self.target},axis=1).columns)

        plt.figure(figsize=(vs,sh),dpi=dpi)
        sns.heatmap(df_comp[:n],annot=True)

    def plot_trees(self,dpi=300,criterion='gini',max_depth=2,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0.0,class_weight=None,ccp_alpha=0.0):

        tree = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion,min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,class_weight=class_weight,ccp_alpha=ccp_alpha)
        tree.fit(self.X_train,self.y_train)

        self.plot_trees_df = pd.DataFrame(index=self.X_train.columns,data=tree.feature_importances_,columns=['Feature Importance']).sort_values('Feature Importance',ascending=False)

        plt.figure(figsize=(12,8),dpi=dpi)
        class_names = [str(cls) for cls in self.y.unique()]
        plot_tree(tree,filled=True,feature_names=self.X_train.columns,proportion=True,rounded=True,precision=2,
            class_names=class_names,label='root',);
        plt.savefig("decision_tree.png")

        return self.plot_trees_df

    def plot_mat(self,estimator_from='voting',estimator=None):   

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel_model

        if estimator_from == 'ada':
            model = self.ada_model

        if estimator_from == 'bagging':
            model = self.bagging_model

        if estimator_from == 'tuning':
            model = self.tuning_model

        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = model.classes_
        ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
        ax.set(xlabel='Predict', ylabel='Actual')

        print(classification_report(self.y_test,y_pred))

    def plot_pca(self,min_n=2,max_n=10):
            
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