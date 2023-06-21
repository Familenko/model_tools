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

    def split_data(self,valid = 0.2,test=0.2):

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

            self.model_list_scaler = scaler

        if mode=='standard':

            self.standard_mark = 'on'
  
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.model_list_scaler = scaler
                

        if mode=='pca':

            self.pca_mark = 'on'

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.pca = PCA(n_components=n_components)
            self.principal_components = self.pca.fit_transform(self.X_train)

            self.X_train = pd.DataFrame(self.principal_components)
            self.X_valid = self.pca.transform(self.X_valid)
            self.X_test = self.pca.transform(self.X_test)

            self.model_list_pca = self.pca

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

    def cv_results(self,model,check_from='ensemble',result='df',param=None):

        if check_from=='ensemble':
            model = self.ensemble_models[model]

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
        
        
        result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[accuracy_valid,precision_valid,
            recall_valid,f1_valid,accuracy_test,precision_test,recall_test,f1_test]},
            index=['accuracy_valid','precision_valid','recall_valid','f1_valid','accuracy_test','precision_test','recall_test','f1_test'])
        
        result_test_df = result_test_df.transpose()
        
        return result_test_df

    def ada(self,n_estimators=50,learning_rate=1.0,estimator='logic'):

        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
        'random':RandomForestClassifier(),'gradient':GradientBoostingClassifier(),'voting':self.voting_clf}

        weak_estimator = models_dic[estimator]

        try:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        except:
            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME.R', n_estimators=n_estimators, learning_rate=learning_rate)
            self.ada_model.fit(self.X_train, self.y_train)

        return self.result_test_df(self.ada_model)

    def model_choose(self,mode='empty',estimator='random',params=None,estimators=None,cv=2,test_cv=2,n_jobs=None):

        self.result_df = pd.DataFrame()

        tree = GridSearchCV(DecisionTreeClassifier(),cv=cv,n_jobs=n_jobs,
                            param_grid = {
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'class_weight': [None, 'balanced']})
        
        random = GridSearchCV(RandomForestClassifier(),cv=cv,n_jobs=n_jobs,
                              param_grid = {
                                'n_estimators': [64, 100, 128],
                                'class_weight': [None, 'balanced']})
                      
        gradient = GridSearchCV(GradientBoostingClassifier(),cv=cv,n_jobs=n_jobs,
                                param_grid = {
                                    'n_estimators': [64, 100, 124]})

        logic = GridSearchCV(LogisticRegression(),cv=cv,n_jobs=n_jobs,
                             param_grid = {
                                'penalty': ['l1', 'l2'],
                                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                'solver': ['lbfgs', 'sag', 'saga'],
                                'max_iter': [100, 250, 500],
                                'class_weight': ['balanced', None]})
        
        knn = GridSearchCV(KNeighborsClassifier(),cv=cv,n_jobs=n_jobs,
                           param_grid={
                            'weights':['uniform', 'distance'],
                            'n_neighbors':range(1,30)})
        
        svc = GridSearchCV(SVC(),cv=cv,n_jobs=n_jobs,
                           param_grid={                        
                            'kernel': ['linear', 'rbf'], 
                            'C': [0.1, 1, 10],
                            'gamma': ['scale', 'auto']})

        if mode == 'empty':
        
            tree = GridSearchCV(DecisionTreeClassifier(),cv=cv,param_grid={},n_jobs=n_jobs)
            random = GridSearchCV(RandomForestClassifier(),cv=cv,param_grid={},n_jobs=n_jobs)       
            gradient = GridSearchCV(GradientBoostingClassifier(),cv=cv,param_grid={},n_jobs=n_jobs)
            logic = GridSearchCV(LogisticRegression(),cv=cv,param_grid={},n_jobs=n_jobs)
            knn = GridSearchCV(KNeighborsClassifier(),cv=cv,param_grid={},n_jobs=n_jobs)
            svc = GridSearchCV(SVC(),cv=cv,param_grid={},n_jobs=n_jobs)

        if mode == 'classic' or mode == 'empty':

            self.models = [tree,random,gradient,logic,knn,svc]
            models_names = ['tree','random','gradient','logic','knn','svc']

            for i in range(len(self.models)):
                model = self.models[i]
                model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_valid)
                accuracy = accuracy_score(self.y_valid, y_pred)
                precision = precision_score(self.y_valid, y_pred, average='macro')
                recall = recall_score(self.y_valid, y_pred, average='macro')
                f1 = f1_score(self.y_valid, y_pred, average='macro')
                
                self.model_list.append(model)
                self.pre.append(precision)
                self.acc.append(accuracy)
                self.rec.append(recall)
                self.f.append(f1)
                
                if len(self.y_test.unique())>1:

                    precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision_macro',cv=test_cv)
                    accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                    recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall_macro',cv=test_cv)
                    f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1_macro',cv=test_cv)

                else:

                    precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision',cv=test_cv)
                    accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                    recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall',cv=test_cv)
                    f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1',cv=test_cv)               

                new_df = pd.DataFrame({models_names[i]:[accuracy,precision,recall,f1,
                                                        accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean(),
                                                        model.best_estimator_.get_params()]},index=['accuracy','precision','recall','f1',
                                                                                                    'accuracy_check','precision_check','recall_check','f1_check',
                                                                                                    'parameters'])

                self.result_df = pd.concat([self.result_df, new_df], axis=1)

        if mode == 'one':

            tree = GridSearchCV(DecisionTreeClassifier(),cv=cv,param_grid = {**params},n_jobs=n_jobs)        
            random = GridSearchCV(RandomForestClassifier(),cv=cv,param_grid = {**params},n_jobs=n_jobs)
            gradient = GridSearchCV(GradientBoostingClassifier(),cv=cv,param_grid = {**params},n_jobs=n_jobs)
            logic = GridSearchCV(LogisticRegression(),cv=cv,param_grid = {**params},n_jobs=n_jobs)
            knn = GridSearchCV(KNeighborsClassifier(),cv=cv,param_grid={**params},n_jobs=n_jobs)
            svc = GridSearchCV(SVC(),cv=cv,param_grid={**params},n_jobs=n_jobs)

            self.models = {'tree':tree,'random':random,'gradient':gradient,'logic':logic,'knn':knn,'svc':svc}
            models_names = estimator

            model = self.models[estimator]
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_valid)
            accuracy = accuracy_score(self.y_valid, y_pred)
            precision = precision_score(self.y_valid, y_pred, average='macro')
            recall = recall_score(self.y_valid, y_pred, average='macro')
            f1 = f1_score(self.y_valid, y_pred, average='macro')
            
            if len(self.y_test.unique())>1:

                precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision_macro',cv=test_cv)
                accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall_macro',cv=test_cv)
                f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1_macro',cv=test_cv)

            else:

                precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision',cv=test_cv)
                accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall',cv=test_cv)
                f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1',cv=test_cv)               

            new_df = pd.DataFrame({estimator:[accuracy,precision,recall,f1,
                                                    accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean(),
                                                    model.best_estimator_.get_params()]},index=['accuracy','precision','recall','f1',
                                                                                                'accuracy_check','precision_check','recall_check','f1_check',
                                                                                                'parameters'])

            self.result_df = pd.concat([self.result_df, new_df], axis=1)

        if mode == 'list':

            self.models = {'tree':tree,'random':random,'gradient':gradient,'logic':logic,'knn':knn,'svc':svc}

            for i in estimators:
                model = self.models[i]

                model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_valid)
                accuracy = accuracy_score(self.y_valid, y_pred)
                precision = precision_score(self.y_valid, y_pred, average='macro')
                recall = recall_score(self.y_valid, y_pred, average='macro')
                f1 = f1_score(self.y_valid, y_pred, average='macro')
                
                self.model_list.append(model)
                self.pre.append(precision)
                self.acc.append(accuracy)
                self.rec.append(recall)
                self.f.append(f1)
                
                if len(self.y_test.unique())>1:

                    precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision_macro',cv=test_cv)
                    accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                    recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall_macro',cv=test_cv)
                    f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1_macro',cv=test_cv)

                else:

                    precision_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='precision',cv=test_cv)
                    accuracy_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='accuracy',cv=test_cv)
                    recall_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='recall',cv=test_cv)
                    f1_check = cross_validate(model,self.X_test,self.y_test,return_train_score=True,scoring='f1',cv=test_cv)               

                new_df = pd.DataFrame({i:[accuracy,precision,recall,f1,
                                                        accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean(),
                                                        model.best_estimator_.get_params()]},index=['accuracy','precision','recall','f1',
                                                                                                    'accuracy_check','precision_check','recall_check','f1_check',
                                                                                                    'parameters'])

                self.result_df = pd.concat([self.result_df, new_df], axis=1)

        self.result_df = self.result_df.transpose()
        self.result_df['overlearn'] = self.result_df['f1'] / self.result_df['f1_check']

        return self.result_df.sort_values('overlearn')

    def auto_build(self,cv=2,test_cv=2,test=0.2,n_jobs=None):
        
        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
                     'tree':DecisionTreeClassifier(),'random':RandomForestClassifier(),
                      'gradient':GradientBoostingClassifier()}
        
        result_rer = self.result_df
        result_rer = result_rer.reset_index()
        best_res_model = result_rer[result_rer['overlearn']<1.3].sort_values('f1').iloc[-1]
        model_index = best_res_model['index']
        model_param = best_res_model['parameters']
        params_with_brackets = {key: [value] for key, value in model_param.items()}
        
        X = self.df.drop({self.target},axis=1)
        y = self.df[self.target]

        X, X_check, y, y_check = train_test_split(X, y, test_size=0.1)
        self.X_train_b, self.X_valid_b, self.y_train_b, self.y_valid_b = train_test_split(X, y, test_size=test)
        
        if self.standard_mark == 'on':
  
            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

        if self.minmax_mark == 'on':
  
            scaler = MinMaxScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

        if self.pca_mark == 'on':

            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

            self.build_pca = PCA(n_components=self.pca_n_components)
            self.X_train_b_pca = self.build_pca.fit_transform(self.X_train_b)

            self.X_train_b = pd.DataFrame(self.X_train_b_pca)
            self.X_valid_b = self.build_pca.transform(self.X_valid_b)
        
        choosen_model = models_dic[model_index]
        model = GridSearchCV(choosen_model,param_grid=params_with_brackets,cv=cv,n_jobs=n_jobs)
        model.fit(self.X_train_b, self.y_train_b)
        
        y_pred = model.predict(self.X_valid_b)
        accuracy = accuracy_score(self.y_valid_b, y_pred)
        precision = precision_score(self.y_valid_b, y_pred, average='macro')
        recall = recall_score(self.y_valid_b, y_pred, average='macro')
        f1 = f1_score(self.y_valid_b, y_pred, average='macro')

        if len(self.y_test.unique())>1:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision_macro',cv=test_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=test_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall_macro',cv=test_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1_macro',cv=test_cv)

        else:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision',cv=test_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=test_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall',cv=test_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1',cv=test_cv)  
        
        self.build_df = pd.DataFrame({'build_result':[accuracy,precision,recall,f1,precision_check['test_score'].mean(),accuracy_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean()]},
                                               index=['accuracy','precision','recall','f1','precision_check','accuracy_check','recall_check','f1_check'])
        
        self.build_df = self.build_df.transpose()
        self.build_df['overlearn'] = self.build_df['f1'] / self.build_df['f1_check']

        self.build_model = model
        self.build_scaler = scaler
        
        return self.build_df

    def choose_build(self,model_index,mode='auto',params=None,cv=2,test=0.2,test_cv=2,n_jobs=None):
        
        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
                     'tree':DecisionTreeClassifier(),'random':RandomForestClassifier(),
                      'gradient':GradientBoostingClassifier()}
        
        if mode == 'auto':

            result_rer = self.result_df
            model_param = result_rer.loc[model_index]['parameters']
            params_with_brackets = {key: [value] for key, value in model_param.items()}
        
        X = self.df.drop({self.target},axis=1)
        y = self.df[self.target]

        X, X_check, y, y_check = train_test_split(X, y, test_size=0.1)
        self.X_train_b, self.X_valid_b, self.y_train_b, self.y_valid_b = train_test_split(X, y, test_size=test)
        
        if self.standard_mark == 'on':
  
            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

        if self.minmax_mark == 'on':
  
            scaler = MinMaxScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

        if self.pca_mark == 'on':

            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_valid_b = scaler.transform(self.X_valid_b)

            self.build_pca = PCA(n_components=self.pca_n_components)
            self.X_train_b_pca = self.build_pca.fit_transform(self.X_train_b)

            self.X_train_b = pd.DataFrame(self.X_train_b_pca)
            self.X_valid_b = self.build_pca.transform(self.X_valid_b)

        if mode == 'auto':

            choosen_model = models_dic[model_index]
            model = GridSearchCV(choosen_model,param_grid=params_with_brackets,cv=cv,n_jobs=n_jobs)

        if mode == 'set':

            choosen_model = models_dic[model_index]
            choosen_model.set_params(**params)
            model = GridSearchCV(choosen_model,param_grid={},cv=cv)

        model.fit(self.X_train_b, self.y_train_b)
        
        y_pred = model.predict(self.X_valid_b)
        accuracy = accuracy_score(self.y_valid_b, y_pred)
        precision = precision_score(self.y_valid_b, y_pred, average='macro')
        recall = recall_score(self.y_valid_b, y_pred, average='macro')
        f1 = f1_score(self.y_valid_b, y_pred, average='macro')
        
        if len(self.y_test.unique())>1:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision_macro',cv=test_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=test_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall_macro',cv=test_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1_macro',cv=test_cv)

        else:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision',cv=test_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=test_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall',cv=test_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1',cv=test_cv)  
        
        self.build_df = pd.DataFrame({'build_result':[accuracy,precision,recall,f1,precision_check['test_score'].mean(),accuracy_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean()]},
                                               index=['accuracy','precision','recall','f1','precision_check','accuracy_check','recall_check','f1_check'])
        
        self.build_df = self.build_df.transpose()
        self.build_df['overlearn'] = self.build_df['f1'] / self.build_df['f1_check']

        self.build_model = model
        self.build_scaler = scaler
        
        return self.build_df
        
    def get_build(self):

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_model.best_estimator_)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_pca,self.build_model.best_estimator_)

        return self.build_pipe

    def get_build_model_list(self,index=-1):

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.model_list_build_pipe = make_pipeline(self.model_list_scaler,self.model_list[index].best_estimator_)

        if self.pca_mark == 'on':
            self.model_list_build_pipe = make_pipeline(self.model_list_scaler,self.model_list_pca,self.model_list[index].best_estimator_)

        return self.model_list_build_pipe

    def plot_mat(self,mode='voting'):   

        if mode == 'voting':
            model = self.voting_model
        if mode == 'ada':
            model = self.ada_model

        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = model.classes_
        ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
        ax.set(xlabel='Predict', ylabel='Actual')

        print(classification_report(self.y_test,y_pred))
 
    def pca_heat(self,n=100,vs=18,sh=4,dpi=150):

        df_comp = pd.DataFrame(self.build_pca.components_,columns=self.df.drop({self.target},axis=1).columns)

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

    def get_build_ada(self):

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.ada)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_pca,self.ada)

        return self.build_pipe
