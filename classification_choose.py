from tqdm import tqdmimport timeitfrom sklearn.linear_model import LogisticRegressionfrom sklearn.neighbors import KNeighborsClassifierfrom sklearn.svm import SVCfrom xgboost import XGBClassifierfrom sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,    AdaBoostClassifier,VotingClassifier, BaggingClassifier, ExtraTreesClassifier)from sklearn.tree import DecisionTreeClassifier, plot_treefrom sklearn.model_selection import GridSearchCV, RandomizedSearchCVfrom sklearn.base import BaseEstimator, ClassifierMixinimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as snsimport numpy as npfrom sklearn.model_selection import train_test_splitfrom sklearn.preprocessing import StandardScaler, MinMaxScalerfrom sklearn.decomposition import PCAfrom sklearn.pipeline import make_pipelinefrom sklearn.metrics import (confusion_matrix, classification_report,    accuracy_score, precision_score,recall_score, f1_score)from scikitplot.metrics import plot_precision_recallclass classifier_choose():    def info(self):        print(f'''        Posible estimator :         ['ExtraTreesClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier',  'SVC',        'GradientBoostingClassifier', 'XGBClassifier', 'LogisticRegression', 'KNeighborsClassifier'        Posible tuner :         'GridSearchCV', 'RandomizedSearchCV', 'default'        Posible metric :         'accuracy', 'precision', 'recall', 'f1'        ''')        def __init__(self,target,df,class_weight=None,probability=True):        self.df = df        self.target = target        self.pca_mark = 'off'        self.standard_mark = 'off'        self.minmax_mark = 'off'                self.X = self.df.drop({self.target},axis=1)        self.y = self.df[self.target]        self.models_dic_base = {        'ExtraTreesClassifier':ExtraTreesClassifier(class_weight=class_weight),        'DecisionTreeClassifier':DecisionTreeClassifier(class_weight=class_weight),        'RandomForestClassifier':RandomForestClassifier(class_weight=class_weight),        'GradientBoostingClassifier':GradientBoostingClassifier(),        'XGBClassifier':XGBClassifier(),'SVC':SVC(probability=probability),        'LogisticRegression':LogisticRegression(class_weight=class_weight,max_iter=200),        'KNeighborsClassifier':KNeighborsClassifier()}        tree_dic_params = {'min_samples_split': [2, 3, 4],'min_samples_leaf': [1, 2, 3],'max_features': ['sqrt', 'log2']}        tree_dic_params_reg = {'min_weight_fraction_leaf':[0.0, 0.1], 'min_impurity_decrease': [0.0, 0.1],        'ccp_alpha': [0.0, 0.1],'max_depth': [3, 10, 100]}        self.models_dic_grid = {            ExtraTreesClassifier(class_weight=class_weight) : {'n_estimators' : [64, 100, 128]}|tree_dic_params,            DecisionTreeClassifier(class_weight=class_weight) : {'splitter' : ["best", "random"]}|tree_dic_params,            RandomForestClassifier(class_weight=class_weight) : {'n_estimators': [64, 100, 128]}|tree_dic_params,            GradientBoostingClassifier() : {'n_estimators': [64, 100, 128]}|tree_dic_params,            XGBClassifier() :{'eta': np.logspace(np.log10(0.1), np.log10(3.0), num=3),'min_split_loss': [ 0.1, 0.2, 0.3],            'alpha': [ 0.01, 0.1, 1],'lambda': [ 0.01, 0.1, 1],'min_child_weight': [1, 5, 10]}|tree_dic_params|tree_dic_params_reg,            SVC(probability=probability) : {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},            LogisticRegression(class_weight=class_weight,max_iter=200,penalty='elasticnet') : {            'C': np.logspace(np.log10(0.01), np.log10(100.0), num=4),'solver': ['lbfgs', 'sag', 'saga']},            KNeighborsClassifier() : {'n_neighbors' : [1,2,4,8,16,32,64], 'weights' : ['uniform', 'distance']}}        self.models_dic_random = {            ExtraTreesClassifier(class_weight=class_weight) : {'n_estimators' : range(64,128)}|tree_dic_params|tree_dic_params_reg,            DecisionTreeClassifier(class_weight=class_weight) : {'splitter' : ["best", "random"]}|tree_dic_params|tree_dic_params_reg,            RandomForestClassifier(class_weight=class_weight) : {'n_estimators': range(64,128)}|tree_dic_params|tree_dic_params_reg,            GradientBoostingClassifier() : {'n_estimators': range(64,128)}|tree_dic_params|tree_dic_params_reg,            XGBClassifier() :{'eta': np.logspace(np.log10(0.1), np.log10(3.0), num=10),'min_split_loss': [ 0.1, 0.2, 0.3],            'alpha': [ 0.01, 0.1, 1],'lambda': [ 0.01, 0.1, 1],'min_child_weight': [1, 5, 10]}|tree_dic_params|tree_dic_params_reg,            SVC(probability=probability) :             {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10],'gamma': ['scale', 'auto']},            LogisticRegression(class_weight=class_weight,max_iter=200) :             {'penalty': ['l1', 'l2','elasticnet'],'C': np.logspace(np.log10(0.01), np.log10(100.0), num=10),            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],            'fit_intercept' : [True,False]},            KNeighborsClassifier() :             {'n_neighbors' : range(1, 64, 1), 'weights' : ['uniform', 'distance']}}    def elapsed_time_decorator(func):        def wrapper(*args, **kwargs):            start_time = timeit.default_timer()            result = func(*args, **kwargs)            elapsed_time = round(timeit.default_timer() - start_time, 2)            print("Elapsed time:", elapsed_time)            return result        return wrapper    def split_data(self,valid = 0.15,test=0.15,stratify=False):        if stratify:            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test, stratify=self.y)            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid,stratify=y)        else:            X, self.X_test, y, self.y_test = train_test_split(self.X, self.y, test_size=test)            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=valid)        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test    @elapsed_time_decorator    def preprocessing(self,mode='StandardScaler',n_components=2):        if mode=='MinMaxScaler':            self.minmax_mark = 'on'            self.standard_mark = 'off'            self.pca_mark = 'off'            scaler = MinMaxScaler()        if mode=='StandardScaler':            self.minmax_mark = 'off'            self.standard_mark = 'on'            self.pca_mark = 'off'            scaler = StandardScaler()        if mode=='MinMaxScaler' or mode=='StandardScaler':            self.X_train = scaler.fit_transform(self.X_train)            self.X_valid = scaler.transform(self.X_valid)            self.X_test = scaler.transform(self.X_test)            self.scaler = scaler                        if mode=='PCA':            self.minmax_mark = 'off'            self.standard_mark = 'off'            self.pca_mark = 'on'            scaler = StandardScaler()            self.X_train = scaler.fit_transform(self.X_train)            self.X_valid = scaler.transform(self.X_valid)            self.X_test = scaler.transform(self.X_test)            pca = PCA(n_components=n_components)            self.X_train = pca.fit_transform(self.X_train)            self.X_valid = pca.transform(self.X_valid)            self.X_test = pca.transform(self.X_test)            self.X_train = pd.DataFrame(self.X_train)            self.X_valid = pd.DataFrame(self.X_valid)            self.X_test = pd.DataFrame(self.X_test)            self.scaler = scaler            self.pca = pca        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test    def preanalize(self,alpha=0.5,bins=20):        # correlation plot        corr_df = pd.DataFrame(self.df).corr()        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)        sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index,                     y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)        ax1.set_title(f"Feature Correlation to {self.target}")        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)        # distribution plot        if len(pd.DataFrame(self.df)[f'{self.target}'].value_counts()) < 10:            cluster_counts = pd.DataFrame(self.df)[f'{self.target}'].value_counts()            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')        else:            sns.histplot(data=pd.DataFrame(self.df), x=f'{self.target}', kde=True, color='green', bins=bins, ax=ax2)        ax2.set_title(f"{self.target} Distribution")        ax2.set_xlabel(f"{self.target}")        ax2.set_ylabel("Count")        plt.show()        # PCA plot        scaler = StandardScaler()        X_train = scaler.fit_transform(self.X)        pca = PCA(n_components=2)        principal_components = pca.fit_transform(X_train)        X_train_pca = pd.DataFrame(principal_components)        print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')        print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')        plt.figure(figsize=(12,6))        sns.scatterplot(x=X_train_pca[0],y=X_train_pca[1],data=pd.DataFrame(self.X),hue=self.y,alpha=alpha)        plt.xlabel('First principal component')        plt.ylabel('Second Principal Component')        plt.show()    @elapsed_time_decorator    def ensemble(self,tuner='default',estimators=None,cv=5,scoring='accuracy',n_iter=10,n_jobs=2,average='macro'):        self.result_df_ensamble = pd.DataFrame()        self.ensemble_models = {}        if tuner == "GridSearchCV":            model_dict_for_tuner = self.models_dic_grid        if tuner == "RandomizedSearchCV" or tuner == 'default':            model_dict_for_tuner = self.models_dic_random        if estimators == 'tree_only':            estimators = list(self.models_dic_base.keys())[:5]        if estimators == 'no_tree':            estimators = list(self.models_dic_base.keys())[5:]        if estimators:            model_dict_for_tuner = {key: model_dict_for_tuner[key] for key,value in model_dict_for_tuner.items()             if key.__class__.__name__ in estimators}        for key, value in tqdm(model_dict_for_tuner.items(), desc="Tuning Ensemble Models"):            ensemble_model_name = key.__class__.__name__            print(f"Model: {ensemble_model_name}")            if tuner == "GridSearchCV":                ensemble_search = GridSearchCV(key,cv=cv,n_jobs=n_jobs,param_grid=value,scoring=scoring)            if tuner == "RandomizedSearchCV":                ensemble_search = RandomizedSearchCV(key,cv=cv,n_jobs=n_jobs,param_distributions=value,scoring=scoring,n_iter=n_iter)            if tuner == 'default':                ensemble_search = GridSearchCV(key,cv=cv,n_jobs=n_jobs,param_grid={},scoring=scoring)            ensemble_search.fit(self.X_train,self.y_train)            self.ensemble_models[ensemble_model_name] = ensemble_search            df_iter_model_ensemble = self.result_test_df(ensemble_search,average)            df_iter_model_ensemble = df_iter_model_ensemble.transpose()            df_iter_model_ensemble.rename(columns={df_iter_model_ensemble.columns[-1]: str(key)}, inplace=True)            self.result_df_ensamble = pd.concat([self.result_df_ensamble, df_iter_model_ensemble], axis=1)        self.result_df_ensamble = self.result_df_ensamble.transpose()        self.result_df_ensamble.index = [index[:index.find('(')] for index in self.result_df_ensamble.index]        fig, axes = plt.subplots()        sns.scatterplot(x=self.result_df_ensamble['precision_valid'], y=self.result_df_ensamble['recall_valid'],             hue = self.result_df_ensamble.index, size=self.result_df_ensamble['f1_valid'], sizes=(50, 200), ax=axes)        axes.set_xlabel('Precision')        axes.set_ylabel('Recall')        axes.legend(loc=(1.1,0.0))        return self.result_df_ensamble.iloc[:, :7]    @elapsed_time_decorator    def voting(self,voting='soft',test_on='valid_data',average='macro'):        self.voting_model = VotingClassifier(estimators=[(key, value.best_estimator_) if hasattr(value, 'best_estimator_')        else (key, value) for key, value in self.ensemble_models.items()],voting=voting)        self.voting_model.fit(self.X_train,self.y_train)        if test_on=='valid_data':            return self.result_test_df(self.voting_model,average).iloc[:, :7]        if test_on=='test_data':            return self.result_test_df(self.voting_model,average)    @elapsed_time_decorator    def basemodel(self,estimator='LogisticRegression',mode='auto_random',params=None,cv=5,scoring='accuracy',        n_iter=10,n_jobs=None,test_on='valid_data',average='macro'):        model = self.models_dic_base[estimator]        if mode == 'set_manual':            params_with_brackets = {key: [value] for key, value in params.items()}            search = GridSearchCV(model,cv=cv,n_jobs=n_jobs,param_grid={**params_with_brackets},scoring=scoring)        if mode == 'set_grid':            search = GridSearchCV(model,cv=cv,n_jobs=n_jobs,param_grid={**params},scoring=scoring)        if mode == 'auto_grid':            index_for_params_grid = {key.__class__.__name__: index for index, key in enumerate(list(self.models_dic_grid.keys()))}            parameter_dic_grid = self.models_dic_grid[list(self.models_dic_grid.keys())[index_for_params_grid[estimator]]]            search = GridSearchCV(model,cv=cv,n_jobs=n_jobs,param_grid=parameter_dic_grid,scoring=scoring)        if mode == 'set_random':            search = RandomizedSearchCV(model,cv=cv,n_jobs=n_jobs,param_distributions={**params},scoring=scoring,n_iter=n_iter)                    if mode == 'auto_random':            index_for_params_random = {key.__class__.__name__: index for index, key in enumerate(list(self.models_dic_random.keys()))}            parameter_dic_random = self.models_dic_random[list(self.models_dic_random.keys())[index_for_params_random[estimator]]]            search = RandomizedSearchCV(model,cv=cv,n_jobs=n_jobs,param_distributions=parameter_dic_random,scoring=scoring,n_iter=n_iter)        search.fit(self.X_train, self.y_train)        self.basemodel_model = search        if test_on=='valid_data':            return self.result_test_df(search,average).iloc[:, :7]        if test_on=='test_data':            return self.result_test_df(search,average)    @elapsed_time_decorator    def tuning(self,estimator_from='default',estimator='KNeighborsClassifier', target='n_neighbors',min_n=1, max_n=30,        step=1,set_target=None,test_on='valid_data',average='macro',params=None,):        if estimator_from == 'default':            tuning_model = self.models_dic_base[estimator]        if estimator_from == 'ensemble':            tuning_model = self.ensemble_models[estimator].best_estimator_            params = self.ensemble_models[estimator].best_params_        if estimator_from == 'basemodel':            tuning_model = self.basemodel_model.best_estimator_            params = self.basemodel_model.best_params_        if estimator_from == 'ada':            tuning_model = self.ada_model            params = self.ada_model.get_params()        if estimator_from == 'bagging':            tuning_model = self.bagging_model            params = self.bagging_model.get_params()        if estimator_from == 'tuning':            tuning_model = self.tuning_model            params = self.tuning_model.get_params()        if estimator_from == 'threshold':            tuning_model = self.threshold_model            params = self.threshold_model.get_params()        if params and target in params:            del params[target]        if set_target:            params_dict = {target : set_target}            if params:                params_dict.update(params)            tuning_model.set_params(**params_dict)            tuning_model.fit(self.X_train, self.y_train)            self.tuning_model = tuning_model            if test_on=='valid_data':                return self.result_test_df(tuning_model,average).iloc[:, :7]            if test_on=='test_data':                return self.result_test_df(tuning_model,average)        else:            metrics = ['precision', 'recall', 'f1','accuracy']            test_error_rates = {metric: [] for metric in metrics}            fig, axs = plt.subplots(2, 2, figsize=(10, 10))            axs = axs.flatten()            for i in tqdm(np.arange(min_n, max_n, step), desc=f"Checking model for {target}"):                try:                    params_dict = {target: i}                    if params:                        params_dict.update(params)                    tuning_model.set_params(**params_dict)                except InvalidParameterError:                    i = i.astype('int')                    params_dict = {target: i}                    if params:                        params_dict.update(params)                    tuning_model.set_params(**params_dict)                tuning_model.fit(self.X_train, self.y_train)                y_pred_valid = tuning_model.predict(self.X_valid)                test_error_rates['accuracy'].append(accuracy_score(self.y_valid, y_pred_valid))                test_error_rates['precision'].append(precision_score(self.y_valid, y_pred_valid,average=average))                test_error_rates['recall'].append(recall_score(self.y_valid, y_pred_valid,average=average))                test_error_rates['f1'].append(f1_score(self.y_valid, y_pred_valid,average=average))            axs[0].plot(np.arange(min_n, max_n, step), test_error_rates['precision'])            axs[0].set_ylabel('precision')            axs[0].set_xlabel(f'{target}')            axs[0].grid(True)            axs[1].plot(np.arange(min_n, max_n, step), test_error_rates['recall'])            axs[1].set_ylabel('recall')            axs[1].set_xlabel(f'{target}')            axs[1].grid(True)            axs[2].plot(np.arange(min_n, max_n, step), test_error_rates['f1'])            axs[2].set_ylabel('f1')            axs[2].set_xlabel(f'{target}')            axs[2].grid(True)            axs[3].plot(np.arange(min_n, max_n, step), test_error_rates['accuracy'])            axs[3].set_ylabel('accuracy')            axs[3].set_xlabel(f'{target}')            axs[3].grid(True)            plt.tight_layout()            plt.show()    @elapsed_time_decorator    def ada(self,estimator_from='default',estimator='DecisionTreeClassifier',n_estimators=50,learning_rate=1.0,        test_on='valid_data',average='macro'):        if estimator_from == 'default':     weak_estimator = self.models_dic_base[estimator]        if estimator_from == 'ensemble':    weak_estimator = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      weak_estimator = self.voting_model        if estimator_from == 'basemodel':   weak_estimator = self.basemodel_model.best_estimator_        if estimator_from == 'tuning':      weak_estimator = self.tuning_model        if estimator_from == 'bagging':     weak_estimator = self.bagging_model        if estimator_from == 'threshold':   weak_estimator = self.threshold_model        try:            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME.R',                 n_estimators=n_estimators, learning_rate=learning_rate)            self.ada_model.fit(self.X_train, self.y_train)        except:            self.ada_model = AdaBoostClassifier(base_estimator=weak_estimator, algorithm='SAMME',                 n_estimators=n_estimators, learning_rate=learning_rate)            self.ada_model.fit(self.X_train, self.y_train)        if test_on=='valid_data':            return self.result_test_df(self.ada_model,average).iloc[:, :7]        if test_on=='test_data':            return self.result_test_df(self.ada_model,average)    @elapsed_time_decorator    def bagging(self,estimator_from='default',estimator='DecisionTreeClassifier',n_estimators=500,max_samples=0.1,        bootstrap=True,n_jobs=1,oob_score=True,max_features=1.0,bootstrap_features=True,test_on='valid_data',average='macro'):        if estimator_from == 'default':     weak_estimator = self.models_dic_base[estimator]        if estimator_from == 'ensemble':    weak_estimator = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      weak_estimator = self.voting_model        if estimator_from == 'basemodel':   weak_estimator = self.basemodel_model.best_estimator_        if estimator_from == 'tuning':      weak_estimator = self.tuning_model        if estimator_from == 'ada':         weak_estimator = self.ada_model        if estimator_from == 'threshold':   weak_estimator = self.threshold_model        self.bagging_model = BaggingClassifier(weak_estimator,n_estimators=n_estimators,max_samples=max_samples,            bootstrap=bootstrap,n_jobs=n_jobs,oob_score=oob_score,max_features=max_features,bootstrap_features=bootstrap_features)        self.bagging_model.fit(self.X_train,self.y_train)        print(f'oob_score - {self.bagging_model.oob_score_}')        if test_on=='valid_data':            return self.result_test_df(self.bagging_model,average).iloc[:, :7]        if test_on=='test_data':            return self.result_test_df(self.bagging_model,average)    def threshold(self,estimator_from='default',estimator='LogisticRegression',set_threshold=None,        test_on='valid_data',average='macro'):        if estimator_from == 'default':     model = self.models_dic_base[estimator]        if estimator_from == 'ensemble':    model = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      model = self.voting_model        if estimator_from == 'basemodel':   model = self.basemodel_model.best_estimator_        if estimator_from == 'tuning':      model = self.tuning_model        if estimator_from == 'bagging':     model = self.bagging_model        if estimator_from == 'ada':         model = self.ada_model        if estimator_from == 'threshold':   model = self.threshold_model        class ThresholdClassifier(BaseEstimator, ClassifierMixin):            def __init__(self, threshold=0.5, estimator=None):                self.threshold = threshold                self.estimator = estimator            def fit(self, X, y):                self.estimator.fit(X, y)                return self            def predict(self, X):                return (self.estimator.predict_proba(X)[:, 1] >= self.threshold).astype(int)            def predict_proba(self, X):                return self.estimator.predict_proba(X)            def get_params(self, deep=True):                return {"threshold": self.threshold, "estimator": self.estimator}            def set_params(self, **parameters):                for parameter, value in parameters.items():                    setattr(self, parameter, value)                return self                        def classes_(self):                return self.estimator.classes_            def __len__(self):                return len(self.estimator.classes_)        def threshold_function(threshold,estimator):            return ThresholdClassifier(threshold,estimator)        if set_threshold:            self.threshold_model = make_pipeline(threshold_function(set_threshold,model))            if test_on=='valid_data':                return self.result_test_df(self.threshold_model,average).iloc[:, :7]            if test_on=='test_data':                return self.result_test_df(self.threshold_model,average)        else:            metrics = ['precision', 'recall', 'f1','accuracy']            fig, axs = plt.subplots(2, 2, figsize=(10, 10))            axs = axs.flatten()            for idx, metric in enumerate(metrics):                test_error_rates = []                for i in tqdm(np.arange(0.01, 0.99, 0.01), desc=f"Threshold assessment for {metric}"):                    threshold_model = make_pipeline(threshold_function(i, model))                    y_pred_valid = threshold_model.predict(self.X_valid)                    if metric == 'accuracy':                        metric_score = accuracy_score(self.y_valid, y_pred_valid)                    elif metric == 'precision':                        metric_score = precision_score(self.y_valid, y_pred_valid,average=average)                    elif metric == 'recall':                        metric_score = recall_score(self.y_valid, y_pred_valid,average=average)                    elif metric == 'f1':                        metric_score = f1_score(self.y_valid, y_pred_valid,average=average)                    test_error_rates.append(metric_score)                axs[idx].plot(np.arange(0.01, 0.99, 0.01), test_error_rates, label=f'{metric} / threshold ratio')                axs[idx].legend()                axs[idx].set_ylabel(f'{metric}')                axs[idx].set_xlabel(f'threshold')                axs[idx].grid(True)             plt.tight_layout()            plt.show()    def cv_results(self,estimator_from='basemodel',estimator=None,result='df'):        if estimator_from == 'ensemble':            model = self.ensemble_models[estimator]        if estimator_from == 'basemodel':            model = self.basemodel_model        results = pd.DataFrame(model.cv_results_)        parameter_names = list(results['params'][0].keys())        parameter_names = ['param_' + param for param in parameter_names]        parameter_names.append('mean_test_score')        parameter_names.append('std_test_score')        parameter_names.append('params')        results.sort_values(by='mean_test_score', ascending=False, inplace=True)        results.reset_index(drop=True, inplace=True)        if result == 'df':            return results[parameter_names]        if result == 'plot':            results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']], subplots=True)            plt.ylabel('Mean test score')            plt.xlabel('Hyperparameter combinations')            plt.grid(True)     def result_test_df(self,model,average='macro'):        y_pred_valid = model.predict(self.X_valid)        accuracy_valid = round(accuracy_score(self.y_valid, y_pred_valid),2)        precision_valid = round(precision_score(self.y_valid, y_pred_valid, average=average),2)        recall_valid = round(recall_score(self.y_valid, y_pred_valid, average=average),2)        f1_valid = round(f1_score(self.y_valid, y_pred_valid, average=average),2)        variance_valid = round(np.var(y_pred_valid),2)        y_pred_test = model.predict(self.X_test)        accuracy_test = round(accuracy_score(self.y_test, y_pred_test),2)        precision_test = round(precision_score(self.y_test, y_pred_test, average=average),2)        recall_test = round(recall_score(self.y_test, y_pred_test,average=average),2)        f1_test = round(f1_score(self.y_test, y_pred_test, average=average),2)        variance_test = round(np.var(y_pred_test),2)        try:            result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[model.best_estimator_.get_params(),                model.cv_results_['mean_fit_time'].sum(),accuracy_valid,precision_valid,recall_valid,f1_valid,                variance_valid,accuracy_test,precision_test,recall_test,f1_test,variance_test]},                index=['parameters','building_time','accuracy_valid','precision_valid','recall_valid','f1_valid',                'variance_valid','accuracy_test','precision_test','recall_test','f1_test','variance_test'])        except AttributeError:            result_test_df = pd.DataFrame({f'{model.__class__.__name__}':[model.get_params(),'|',accuracy_valid,precision_valid,                recall_valid,f1_valid,variance_valid,accuracy_test,precision_test,recall_test,f1_test,variance_test]},                index=['parameters','building_time','accuracy_valid','precision_valid','recall_valid','f1_valid',                'variance_valid','accuracy_test','precision_test','recall_test','f1_test','variance_test'])        result_test_df = result_test_df.transpose()                return result_test_df            def get_pipe(self,estimator_from,estimator=None):        if estimator_from == 'ensemble':    model = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      model = self.voting_model        if estimator_from == 'basemodel':   model = self.basemodel_model.best_estimator_        if estimator_from == 'ada':         model = self.ada_model        if estimator_from == 'bagging':     model = self.bagging_model        if estimator_from == 'tuning':      model = self.tuning_model        if estimator_from == 'threshold':   model = self.threshold_model        if self.standard_mark == 'on' or self.minmax_mark == 'on':            self.build_pipe = make_pipeline(self.scaler,model)        if self.pca_mark == 'on':            self.build_pipe = make_pipeline(self.scaler,self.pca,model)        return self.build_pipe     def heat_pca(self,n=100,vs=18,sh=4,dpi=150):        df_comp = pd.DataFrame(self.pca.components_,columns=self.df.drop({self.target},axis=1).columns)        plt.figure(figsize=(vs,sh),dpi=dpi)        sns.heatmap(df_comp[:n],annot=True)    def plot_tree(self,params={'max_depth':3},dpi=300,save=False):        tree = DecisionTreeClassifier()        tree.set_params(**params)        tree.fit(self.X_train,self.y_train)        self.plot_tree_df = pd.DataFrame(index=self.X_train.columns,data=np.round(tree.feature_importances_,2),            columns=['Feature Importance']).sort_values('Feature Importance',ascending=False)        plt.figure(figsize=(12,8),dpi=dpi)        class_names = [str(cls) for cls in self.y.unique()]        plot_tree(tree,filled=True,feature_names=self.X_train.columns,proportion=True,rounded=True,precision=2,            class_names=class_names,label='root',);        if save:            plt.savefig("plot_tree.png")        return self.plot_tree_df.transpose()    def plot_mat(self,estimator_from='voting',estimator=None,test_on='valid_data'):           if estimator_from == 'ensemble':    model = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      model = self.voting_model        if estimator_from == 'basemodel':   model = self.basemodel_model.best_estimator_        if estimator_from == 'ada':         model = self.ada_model        if estimator_from == 'bagging':     model = self.bagging_model        if estimator_from == 'tuning':      model = self.tuning_model        if estimator_from == 'threshold':   model = self.threshold_model        if test_on=='valid_data':            X = self.X_valid            y = self.y_valid        if test_on=='test_data':            X = self.X_test            y = self.y_test        y_pred = model.predict(X)        cm = confusion_matrix(y, y_pred)        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        classes = model.classes_        try:            ax = sns.heatmap(np.round(cm,2), annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')            ax.set(xlabel='Predict', ylabel='Actual')        except TypeError:            ax = sns.heatmap(np.round(cm,2), annot=True,cmap='Greens')            ax.set(xlabel='Predict', ylabel='Actual')        print(classification_report(y,y_pred))    def plot_precision_recall(self,estimator_from='basemodel',estimator=None,test_on='valid_data'):        if estimator_from == 'ensemble':    model = self.ensemble_models[estimator].best_estimator_        if estimator_from == 'voting':      model = self.voting_model        if estimator_from == 'basemodel':   model = self.basemodel_model.best_estimator_        if estimator_from == 'ada':         model = self.ada_model        if estimator_from == 'bagging':     model = self.bagging_model        if estimator_from == 'tuning':      model = self.tuning_model        if estimator_from == 'threshold':   model = self.threshold_model        if test_on=='valid_data':            X = self.X_valid            y = self.y_valid        if test_on=='test_data':            X = self.X_test            y = self.y_test        plot_precision_recall(y, model.predict_proba(X))        plt.tight_layout()        plt.show()    def plot_pca(self,min_n=2,max_n=10):                    scaler = StandardScaler()        pca_X = scaler.fit_transform(self.X)        explained_variance = []        for n in range(min_n,max_n):            pca = PCA(n_components=n)            pca.fit(pca_X)                        explained_variance.append(np.sum(pca.explained_variance_ratio_))        plt.plot(range(min_n,max_n),explained_variance)        plt.xlabel("Number of Components")        plt.ylabel("Variance Explained")        plt.grid(alpha=0.2);