import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
    
    def __init__(self,target,df,procent=1.0):
        
        self.df = df
        self.target = target

        self.pca_mark = 'off'
        self.standard_mark = 'off'
        self.minmax_mark = 'off'
        
        self.procented_df = df.sample(frac=1).iloc[:int(len(df.index)*procent)]
        self.X = self.procented_df.drop({self.target},axis=1)
        self.y = self.procented_df[self.target]

        indexes_to_drop = self.procented_df.index.tolist()
        self.without_procented_df = df.drop(indexes_to_drop)

        self.models_dic_base = {'LogisticRegression':LogisticRegression(),'KNeighborsClassifier':KNeighborsClassifier(),'SVC':SVC(),
        'RandomForestClassifier':RandomForestClassifier(),'GradientBoostingClassifier':GradientBoostingClassifier()}

    def split_data(self,valid = 0.15,test=0.15):

        X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid)

    def preprocessing(self,mode='standard',n_components=2):

        self.pca_n_components=n_components

        if mode=='minmax':

            self.standard_mark = 'on'
  
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.scaler = scaler

        if mode=='standard':

            self.standard_mark = 'on'
  
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.scaler = scaler
                
        if mode=='pca':

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

            self.pca = pca

    def preanalize(self,alpha=0.5):

        # correlation plot
        corr_df = pd.DataFrame(self.procented_df).corr()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

        sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index, 
                    y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title(f"Feature Correlation to {self.target}")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        # distribution plot
        if len(pd.DataFrame(self.procented_df)[f'{self.target}'].value_counts()) < 10:
            cluster_counts = pd.DataFrame(self.procented_df)[f'{self.target}'].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.histplot(data=pd.DataFrame(self.procented_df), x=f'{self.target}', kde=True, color='green', bins=20, ax=ax2)

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

    def ensemble(self,tuner='RandomizedSearchCV',cv=5,n_jobs=None,scoring='accuracy',random_iter=5,class_weight=None):

        self.result_df_ensamble = pd.DataFrame()
        self.ensemble_models = {}

        model_dict_for_grid = {

            RandomForestClassifier() : {'n_estimators': [64, 100, 128],'class_weight': [class_weight]},
            GradientBoostingClassifier() : {'n_estimators': [64, 100, 128]},
            LogisticRegression() : {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(0.01,100, 10),'solver': ['lbfgs', 'sag', 'saga'],'class_weight': [class_weight]},
            SVC() : {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},
            KNeighborsClassifier() : {'n_neighbors' : [1,2,4,8,16,32], 'weights' : ['uniform', 'distance']}}

        model_dict_for_random = {

            RandomForestClassifier() : 
            {'n_estimators': range(64, 128, 1),'class_weight': [class_weight]},

            GradientBoostingClassifier() : 
            {'n_estimators': range(64, 128, 1)},

            LogisticRegression() : 
            {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(0.01,100, 100),
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'class_weight': [class_weight],'fit_intercept' : [True,False], 'max_iter' : [100,200,500]},

            SVC() : 
            {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},

            KNeighborsClassifier() : 
            {'n_neighbors' : range(1, 30, 1), 'weights' : ['uniform', 'distance']}}

        if tuner == "GridSearchCV":
            model_dict_for_tuner = model_dict_for_grid
        if tuner == "RandomizedSearchCV":
            model_dict_for_tuner = model_dict_for_random

        for key,value in model_dict_for_tuner.items():
            ensemble_model_name = key.__class__.__name__
            if tuner == "GridSearchCV":
                ensemble_search = GridSearchCV(key,cv=cv,n_jobs=n_jobs,param_grid=value,scoring=scoring)
            if tuner == "RandomizedSearchCV":
                ensemble_search = RandomizedSearchCV(key,cv=cv,n_jobs=n_jobs,param_distributions=value,scoring=scoring,n_iter=random_iter)

            ensemble_search.fit(self.X_train,self.y_train)
            self.ensemble_models[ensemble_model_name] = ensemble_search

            df_iter_model_ensemble = self.result_test_df(ensemble_search)
            df_iter_model_ensemble = df_iter_model_ensemble.transpose()
            df_iter_model_ensemble.rename(columns={df_iter_model_ensemble.columns[-1]: str(key)}, inplace=True)
            self.result_df_ensamble = pd.concat([self.result_df_ensamble, df_iter_model_ensemble], axis=1)
        self.result_df_ensamble = self.result_df_ensamble.transpose()

        return self.result_df_ensamble.iloc[:, :4]

    def cv_results(self,estimator_from='ensemble',estimator=None,result='df',param=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel

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
        
        
        result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[model.best_estimator_.get_params(),accuracy_valid,precision_valid,
            recall_valid,f1_valid,accuracy_test,precision_test,recall_test,f1_test]},
            index=['parameters','accuracy_valid','precision_valid','recall_valid','f1_valid','accuracy_test','precision_test','recall_test','f1_test'])
        
        result_test_df = result_test_df.transpose()
        
        return result_test_df

    def ada(self,n_estimators=50,learning_rate=1.0,estimator='logic',estimator_from='ensemble'):

        if estimator_from == 'empty':
            weak_estimator = self.models_dic_base[estimator]

        if estimator_from == 'ensemble':
            weak_estimator = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            weak_estimator = self.voting_model

        if estimator_from == 'basemodel':
            weak_estimator = self.basemodel

        try:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        except:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME.R', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        return self.result_test_df(self.ada_model)

    def basemodel(self,mode,model_name,params,cv,scoring,n_iter,n_jobs):

        basemodel = self.models_dic_base[model_name]

        if mode == 'set':
            basemodel.set_params(**params)

        if mode == 'grid':
            basemodel = GridSearchCV(basemodel,cv=cv,n_jobs=n_jobs,param_grid={**params},scoring=scoring)

        if mode == 'random':
            basemodel = RandomizedSearchCV(basemodel,cv=cv,n_jobs=n_jobs,param_distributions={**params},scoring=scoring,n_iter=random_iter)

        basemodel.fit(self.X_train, self.y_train)
        self.basemodel = basemodel

        return self.result_test_df(basemodel)
        
    def get_pipe(self,estimator_from,estimator=None):

        if estimator_from == 'ensemble':
            model = self.ensemble_models[estimator]

        if estimator_from == 'voting':
            model = self.voting_model

        if estimator_from == 'basemodel':
            model = self.basemodel

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
            model = self.basemodel

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

    def pca_choose(self,min_n=1,max_n=10):
            
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
