class classifier_choose():
    
    def __init__(self,target,df,procent=1.0):
        
        self.df = df
        self.target = target

        self.pca_mark = 'off'
        self.standard_mark = 'off'
        self.minmax_mark = 'off'
        
        self.pre=[]
        self.acc=[]
        self.rec=[]
        self.f=[]
        self.model_list=[]
        
        self.procented_features = df.sample(frac=1).iloc[:int(len(df.index)*procent)]
        
        self.X = self.procented_features.drop({self.target},axis=1)
        self.y = self.procented_features[self.target]

    def split_data(self,valid = 0.2,test=0.1):

        from sklearn.model_selection import train_test_split
        self.X, self.X_check, self.y, self.y_check = train_test_split(self.X, self.y, test_size=test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=valid)

    def preprocessing(self,mode='standard',n_components=2,alpha=0.5):

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        self.pca_n_components=n_components

        if mode=='minmax':

            self.standard_mark = 'on'
  
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.X_check = scaler.transform(self.X_check)

            self.model_list_scaler = scaler

        if mode=='standard':

            self.standard_mark = 'on'
  
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.X_check = scaler.transform(self.X_check)

            self.model_list_scaler = scaler
                

        if mode=='pca':

            self.pca_mark = 'on'

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.X_check = scaler.transform(self.X_check)

            self.pca = PCA(n_components=n_components)
            self.principal_components = self.pca.fit_transform(self.X_train)

            self.X_train = pd.DataFrame(self.principal_components)
            self.X_test = self.pca.transform(self.X_test)
            self.X_check = self.pca.transform(self.X_check)

            self.model_list_pca = self.pca

            if len(self.X_train.columns)==2:

                print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
                print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

                plt.figure(figsize=(8,6))
                sns.scatterplot(x=self.X_train[0],y=self.X_train[1],data=self.df,hue=self.y,alpha=alpha)
                plt.xlabel('First principal component')
                plt.ylabel('Second Principal Component')
                plt.show()

        if mode=='analyze':

            # correlation plot
            corr_df = pd.DataFrame(self.procented_features).corr()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

            sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index, 
                        y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)
            ax1.set_title(f"Feature Correlation to {self.target}")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

            # distribution plot
            if len(pd.DataFrame(self.procented_features)[f'{self.target}'].value_counts()) < 10:
                cluster_counts = pd.DataFrame(self.procented_features)[f'{self.target}'].value_counts()
                ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
            else:
                sns.histplot(data=pd.DataFrame(self.procented_features), x=f'{self.target}', kde=True, color='green', bins=20, ax=ax2)

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

    def classifier_choose(self,mode='empty',estimator='random',params=None,estimators=None,cv=2,check_cv=2,n_jobs=None):
        
        import warnings
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)

        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import cross_validate
        import pandas as pd

        self.result_df = pd.DataFrame()

        mlp = GridSearchCV(MLPClassifier(), cv=5, n_jobs=n_jobs,
                            param_grid={
                                'hidden_layer_sizes': [(100,10,1)],
                                'activation': ['relu', 'tanh', 'logistic'],
                                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                'solver':['lbfgs', 'sgd', 'adam']})

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
            mlp = GridSearchCV(MLPClassifier(),cv=cv,param_grid={},n_jobs=n_jobs)

        if mode == 'classic' or mode == 'empty':

            self.models = [mlp,tree,random,gradient,logic,knn,svc]
            models_names = ['mlp','tree','random','gradient','logic','knn','svc']

            for i in range(len(self.models)):
                model = self.models[i]
                model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='macro')
                recall = recall_score(self.y_test, y_pred, average='macro')
                f1 = f1_score(self.y_test, y_pred, average='macro')
                
                self.model_list.append(model)
                self.pre.append(precision)
                self.acc.append(accuracy)
                self.rec.append(recall)
                self.f.append(f1)
                
                if len(self.y_check.unique())>1:

                    precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
                    accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                    recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
                    f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

                else:

                    precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision',cv=check_cv)
                    accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                    recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall',cv=check_cv)
                    f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1',cv=check_cv)               

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
            mlp = GridSearchCV(MLPClassifier(),cv=cv,param_grid={**params},n_jobs=n_jobs)

            self.models = {'mlp':mlp,'tree':tree,'random':random,'gradient':gradient,'logic':logic,'knn':knn,'svc':svc}
            models_names = estimator

            model = self.models[estimator]
            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')
            f1 = f1_score(self.y_test, y_pred, average='macro')
            
            if len(self.y_check.unique())>1:

                precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
                accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
                f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

            else:

                precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision',cv=check_cv)
                accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall',cv=check_cv)
                f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1',cv=check_cv)               

            new_df = pd.DataFrame({estimator:[accuracy,precision,recall,f1,
                                                    accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean(),
                                                    model.best_estimator_.get_params()]},index=['accuracy','precision','recall','f1',
                                                                                                'accuracy_check','precision_check','recall_check','f1_check',
                                                                                                'parameters'])

            self.result_df = pd.concat([self.result_df, new_df], axis=1)

        if mode == 'list':

            self.models = {'mlp':mlp,'tree':tree,'random':random,'gradient':gradient,'logic':logic,'knn':knn,'svc':svc}

            for i in estimators:
                model = self.models[i]

                model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='macro')
                recall = recall_score(self.y_test, y_pred, average='macro')
                f1 = f1_score(self.y_test, y_pred, average='macro')
                
                self.model_list.append(model)
                self.pre.append(precision)
                self.acc.append(accuracy)
                self.rec.append(recall)
                self.f.append(f1)
                
                if len(self.y_check.unique())>1:

                    precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
                    accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                    recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
                    f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

                else:

                    precision_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='precision',cv=check_cv)
                    accuracy_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
                    recall_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='recall',cv=check_cv)
                    f1_check = cross_validate(model,self.X_check,self.y_check,return_train_score=True,scoring='f1',cv=check_cv)               

                new_df = pd.DataFrame({i:[accuracy,precision,recall,f1,
                                                        accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean(),
                                                        model.best_estimator_.get_params()]},index=['accuracy','precision','recall','f1',
                                                                                                    'accuracy_check','precision_check','recall_check','f1_check',
                                                                                                    'parameters'])

                self.result_df = pd.concat([self.result_df, new_df], axis=1)

        self.result_df = self.result_df.transpose()
        self.result_df['overlearn'] = self.result_df['f1'] / self.result_df['f1_check']

        return self.result_df.sort_values('overlearn')

    def metric_scatter(self):

        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots()
        sns.scatterplot(x=self.pre, y=self.rec, hue=self.models, size=self.acc, sizes=(50, 200), ax=axes)
        axes.set_xlabel('Precision')
        axes.set_ylabel('Recall')
        axes.legend(loc=(1.1,0.0))

    def auto_build(self,cv=2,check_cv=2,test=0.2,n_jobs=None):

        import warnings
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_validate
        
        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
                     'tree':DecisionTreeClassifier(),'random':RandomForestClassifier(),
                      'gradient':GradientBoostingClassifier(),'mlp':MLPClassifier(max_iter=1000)}
        
        result_rer = self.result_df
        result_rer = result_rer.reset_index()
        best_res_model = result_rer[result_rer['overlearn']<1.3].sort_values('f1').iloc[-1]
        model_index = best_res_model['index']
        model_param = best_res_model['parameters']
        params_with_brackets = {key: [value] for key, value in model_param.items()}
        
        X = self.df.drop({self.target},axis=1)
        y = self.df[self.target]

        X, X_check, y, y_check = train_test_split(X, y, test_size=0.1)
        self.X_train_b, self.X_test_b, self.y_train_b, self.y_test_b = train_test_split(X, y, test_size=test)
        
        if self.standard_mark == 'on':
  
            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

        if self.minmax_mark == 'on':
  
            scaler = MinMaxScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

        if self.pca_mark == 'on':

            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

            self.build_pca = PCA(n_components=self.pca_n_components)
            self.X_train_b_pca = self.build_pca.fit_transform(self.X_train_b)

            self.X_train_b = pd.DataFrame(self.X_train_b_pca)
            self.X_test_b = self.build_pca.transform(self.X_test_b)
        
        choosen_model = models_dic[model_index]
        model = GridSearchCV(choosen_model,param_grid=params_with_brackets,cv=cv,n_jobs=n_jobs)
        model.fit(self.X_train_b, self.y_train_b)
        
        y_pred = model.predict(self.X_test_b)
        accuracy = accuracy_score(self.y_test_b, y_pred)
        precision = precision_score(self.y_test_b, y_pred, average='macro')
        recall = recall_score(self.y_test_b, y_pred, average='macro')
        f1 = f1_score(self.y_test_b, y_pred, average='macro')

        if len(self.y_check.unique())>1:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

        else:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1',cv=check_cv)  
        
        self.build_df = pd.DataFrame({'build_result':[accuracy,precision,recall,f1,precision_check['test_score'].mean(),accuracy_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean()]},
                                               index=['accuracy','precision','recall','f1','precision_check','accuracy_check','recall_check','f1_check'])
        
        self.build_df = self.build_df.transpose()
        self.build_df['overlearn'] = self.build_df['f1'] / self.build_df['f1_check']

        self.build_model = model
        self.build_scaler = scaler
        
        return self.build_df

    def choose_build(self,model_index,mode='auto',params=None,cv=2,test=0.2,check_cv=2,n_jobs=None):

        import warnings
        warnings.filterwarnings('ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_validate
        
        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
                     'tree':DecisionTreeClassifier(),'random':RandomForestClassifier(),
                      'gradient':GradientBoostingClassifier(),'mlp':MLPClassifier()}
        
        if mode == 'auto':

            result_rer = self.result_df
            model_param = result_rer.loc[model_index]['parameters']
            params_with_brackets = {key: [value] for key, value in model_param.items()}
        
        X = self.df.drop({self.target},axis=1)
        y = self.df[self.target]

        X, X_check, y, y_check = train_test_split(X, y, test_size=0.1)
        self.X_train_b, self.X_test_b, self.y_train_b, self.y_test_b = train_test_split(X, y, test_size=test)
        
        if self.standard_mark == 'on':
  
            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

        if self.minmax_mark == 'on':
  
            scaler = MinMaxScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

        if self.pca_mark == 'on':

            scaler = StandardScaler()
            self.X_train_b = scaler.fit_transform(self.X_train_b)
            self.X_test_b = scaler.transform(self.X_test_b)

            self.build_pca = PCA(n_components=self.pca_n_components)
            self.X_train_b_pca = self.build_pca.fit_transform(self.X_train_b)

            self.X_train_b = pd.DataFrame(self.X_train_b_pca)
            self.X_test_b = self.build_pca.transform(self.X_test_b)

        if mode == 'auto':

            choosen_model = models_dic[model_index]
            model = GridSearchCV(choosen_model,param_grid=params_with_brackets,cv=cv,n_jobs=n_jobs)

        if mode == 'set':

            choosen_model = models_dic[model_index]
            choosen_model.set_params(**params)
            model = GridSearchCV(choosen_model,param_grid={},cv=cv)

        model.fit(self.X_train_b, self.y_train_b)
        
        y_pred = model.predict(self.X_test_b)
        accuracy = accuracy_score(self.y_test_b, y_pred)
        precision = precision_score(self.y_test_b, y_pred, average='macro')
        recall = recall_score(self.y_test_b, y_pred, average='macro')
        f1 = f1_score(self.y_test_b, y_pred, average='macro')
        
        if len(self.y_check.unique())>1:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

        else:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1',cv=check_cv)  
        
        self.build_df = pd.DataFrame({'build_result':[accuracy,precision,recall,f1,precision_check['test_score'].mean(),accuracy_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean()]},
                                               index=['accuracy','precision','recall','f1','precision_check','accuracy_check','recall_check','f1_check'])
        
        self.build_df = self.build_df.transpose()
        self.build_df['overlearn'] = self.build_df['f1'] / self.build_df['f1_check']

        self.build_model = model
        self.build_scaler = scaler
        
        return self.build_df
        
    def get_build(self):

        from sklearn.pipeline import make_pipeline

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_model.best_estimator_)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_pca,self.build_model.best_estimator_)

        return self.build_pipe

    def get_build_model_list(self,index=-1):

        from sklearn.pipeline import make_pipeline

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.model_list_build_pipe = make_pipeline(self.model_list_scaler,self.model_list[index].best_estimator_)

        if self.pca_mark == 'on':
            self.model_list_build_pipe = make_pipeline(self.model_list_scaler,self.model_list_pca,self.model_list[index].best_estimator_)

        return self.model_list_build_pipe

    def plot_mat(self):   
        
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report

        y_pred = self.build_model.predict(self.X_test_b)
        cm = confusion_matrix(self.y_test_b, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = self.build_model.classes_
        ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
        ax.set(xlabel='Predict', ylabel='Actual')

        print(classification_report(self.y_test_b,y_pred))
 
    def pca_heat(self,n=100,vs=18,sh=4,dpi=150):

        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        df_comp = pd.DataFrame(self.build_pca.components_,columns=self.df.drop({self.target},axis=1).columns)

        plt.figure(figsize=(vs,sh),dpi=dpi)
        sns.heatmap(df_comp[:n],annot=True)

    def pca_choose(self,min_n=1,max_n=10):

        from sklearn.decomposition import PCA
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
            
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

    def ada_build(self,n_estimators=50,learning_rate=1.0,check_cv=10,estimator='strong'):

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


        models_dic = {'logic':LogisticRegression(),'knn':KNeighborsClassifier(),'svc':SVC(),
                     'tree':DecisionTreeClassifier(),'random':RandomForestClassifier(),
                      'gradient':GradientBoostingClassifier(),'strong':self.build_model.best_estimator_}

        best_estimator = models_dic[estimator]

        X, X_check, y, y_check = train_test_split(self.X_train_b, self.y_train_b, test_size=0.2)
        X_train, self.X_test_ada, y_train, self.y_test_ada = train_test_split(X, y, test_size=0.2)

        try:
            model = AdaBoostClassifier(base_estimator=best_estimator, n_estimators=n_estimators, learning_rate=learning_rate)

            model.fit(X_train, y_train)

            y_pred = model.predict(self.X_test_ada)
            accuracy = accuracy_score(self.y_test_ada, y_pred)
            precision = precision_score(self.y_test_ada, y_pred, average='macro')
            recall = recall_score(self.y_test_ada, y_pred, average='macro')
            f1 = f1_score(self.y_test_ada, y_pred, average='macro')

        except TypeError:
            model = AdaBoostClassifier(base_estimator=best_estimator,algorithm='SAMME', n_estimators=n_estimators, learning_rate=learning_rate)

            model.fit(X_train, y_train)

            y_pred = model.predict(self.X_test_ada)
            accuracy = accuracy_score(self.y_test_ada, y_pred)
            precision = precision_score(self.y_test_ada, y_pred, average='macro')
            recall = recall_score(self.y_test_ada, y_pred, average='macro')
            f1 = f1_score(self.y_test_ada, y_pred, average='macro')
                
        if len(y_check.unique())>1:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision_macro',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall_macro',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1_macro',cv=check_cv)

        else:

            precision_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='precision',cv=check_cv)
            accuracy_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='accuracy',cv=check_cv)
            recall_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='recall',cv=check_cv)
            f1_check = cross_validate(model,X_check,y_check,return_train_score=True,scoring='f1',cv=check_cv)               

        self.build_df_ada = pd.DataFrame({'ada':[accuracy,precision,recall,f1,
                                                accuracy_check['test_score'].mean(),precision_check['test_score'].mean(),recall_check['test_score'].mean(),f1_check['test_score'].mean()]},
                                                index=['accuracy','precision','recall','f1','accuracy_check','precision_check','recall_check','f1_check'])

        self.ada = model

        self.build_df_ada = self.build_df_ada.transpose()
        self.build_df_ada['overlearn'] = self.build_df_ada['f1'] / self.build_df_ada['f1_check']

        return self.build_df_ada

    def plot_mat_ada(self):   
        
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report

        y_pred = self.ada.predict(self.X_test_ada)
        cm = confusion_matrix(self.y_test_ada, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        classes = self.ada.classes_
        ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
        ax.set(xlabel='Predict', ylabel='Actual')

        print(classification_report(self.y_test_ada,y_pred))

    def get_build_ada(self):

        from sklearn.pipeline import make_pipeline

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.ada)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.build_pca,self.ada)

        return self.build_pipe
