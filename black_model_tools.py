from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV

from scikitplot.metrics import plot_precision_recall

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
    BaggingRegressor,
)

from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression,
    HuberRegressor,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy.stats import binom, norm

from tqdm import tqdm
import timeit

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelTools:

    def decorator_elapsed_time(func):

        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            elapsed_time = round(timeit.default_timer() - start_time, 2)
            print("Elapsed time:", elapsed_time)
            return result

        return wrapper

    @decorator_elapsed_time
    def fine_tuning(

        self,
        estimator_from="basemodel",
        estimator=None,
        average="macro",
        include_test=True):

        if estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_

        if hasattr(model, "warm_start"):
            model.fit(self.X_valid, self.y_valid)

            sentence = ""
            if include_test:
                model.fit(self.X_test, self.y_test)
                sentence = "and test data"

            self.fine_tuning_model = model

            self.estimator_from_dic["fine_tuning"] = self.fine_tuning_model

            print(
                f"Your model has been additionaly trained with validation data {sentence}"
            )

        else:
            print("This model do not have warm start")

    @staticmethod
    def df_convert_to_boolean(df):

        for column in df.columns:
            if set(df[column].unique()) == {0, 1}:
                df[column] = df[column].astype(bool)

        return df

    @decorator_elapsed_time
    def df_delete_rows(

        self, 
        min_column=10, 
        threshold=1.5, 
        inplace=False):

        self.X.reset_index(inplace=True)
        self.y = pd.DataFrame(self.y)
        self.y.reset_index(inplace=True)
        self.df.reset_index(inplace=True)

        result_df = self.df_convert_to_boolean(self.df.copy())
        df_description = result_df.describe()
        total_outliers = 0

        for column in result_df.columns:
            try:
                _, outlier_indices = self.df_detect_outliers(
                    result_df[column], threshold
                )
                outliers_in_column = len(outlier_indices)
                df_description.loc["Outliers", column] = outliers_in_column
                total_outliers += outliers_in_column
                result_df.loc[outlier_indices, column] = np.nan

            except TypeError:
                continue

        outlier_to_delete = []
        for index, row in result_df.iterrows():
            if row.isnull().sum() >= min_column:
                outlier_to_delete.append(index)

        if not inplace:
            print(
                f"Possible to delete {len(outlier_to_delete)} \
                rows wich have minimum {min_column} outliers each \
                ({round((len(outlier_to_delete) / len(result_df.index)) * 100, 2)}%)")

            return df_description

        if inplace:
            self.X = self.X.drop(outlier_to_delete, axis=0)
            self.y = self.y.drop(outlier_to_delete, axis=0)

            self.X.set_index("index", inplace=True)
            self.y.set_index("index", inplace=True)

            columns_to_keep = self.df.columns.drop(self.target)
            self.df = pd.DataFrame(data=self.X, columns=columns_to_keep)
            self.df[self.target] = self.y
            self.df.drop("index", axis=1, inplace=True)

            print(f"Deleted {len(result_df.index) - len(self.df.index)} rows")

            return self.df

    @decorator_elapsed_time
    def df_fill_outliers_or_na(

        self, 
        threshold=1.5, 
        inplace=False, 
        fill="mean", 
        interpolate_method="linear"):

        self.df.reset_index(inplace=True)

        result_df = self.df_convert_to_boolean(self.df.copy())
        df_description = result_df.describe()
        total_outliers = 0

        for column in result_df.columns:
            try:
                _, outlier_indices = self.df_detect_outliers(
                    result_df[column], threshold
                )
                outliers_in_column = len(outlier_indices)
                df_description.loc["Outliers", column] = outliers_in_column
                total_outliers += outliers_in_column

                result_df.loc[outlier_indices, column] = np.nan

                fillna_dic = {
                    "mean": result_df[column].fillna(result_df[column].mean()),
                    "median": result_df[column].fillna(result_df[column].median()),
                    "interpolate": result_df[column].interpolate(
                        method=interpolate_method)}

                try:
                    fillna_dic["mode"] = result_df[column].fillna(
                        result_df[column].mode()[0])
                    result_df[column] = fillna_dic[fill]

                except KeyError:
                    result_df[column] = fillna_dic[fill]

            except TypeError:
                continue

        if not inplace:
            total_amount = len(result_df.index) * len(result_df.columns)
            print(
                f"Possible to fix {total_outliers} outliers from {total_amount} \
                total amount, ({round((total_outliers / total_amount) * 100, 2)}%)")

            return df_description

        if inplace:
            self.df = result_df
            self.df.set_index("index", inplace=True)
            print(f"{total_outliers} outliers get {fill} column value")

            self.X = self.df.drop({self.target}, axis=1)
            self.y = self.df[self.target]

            return self.df

    def df_detect_outliers(

        self, 
        df, 
        threshold=1.5):

        q1 = np.percentile(df, 25)
        q3 = np.percentile(df, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = []
        outlier_indices = []

        for i, value in enumerate(df):
            if value < lower_bound or value > upper_bound:
                outliers.append(value)
                outlier_indices.append(i)

        return outliers, outlier_indices

    def cv_results(

        self, 
        estimator_from="basemodel", 
        estimator=None, 
        result="df"):

        if estimator_from == "ensemble":
            model = self.ensemble_models[estimator]
        if estimator_from == "basemodel":
            model = self.basemodel_model

        results = pd.DataFrame(model.cv_results_)
        parameter_names = list(results["params"][0].keys())
        parameter_names = ["param_" + param for param in parameter_names]
        parameter_names.append("mean_test_score")
        parameter_names.append("std_test_score")
        parameter_names.append("params")
        results.sort_values(by="mean_test_score", ascending=False, inplace=True)
        results.reset_index(drop=True, inplace=True)

        if result == "df":
            return results[parameter_names]

        if result == "plot":
            results["mean_test_score"].plot(
                yerr=[results["std_test_score"], results["std_test_score"]],
                subplots=True,
            )
            plt.ylabel("Mean test score")
            plt.xlabel("Hyperparameter combinations")
            plt.grid(True)

    def heat_pca(

        self, 
        n=100, 
        vs=18, 
        sh=4, 
        dpi=150):

        df_comp = pd.DataFrame(
            self.pca.components_, columns=self.df.drop({self.target}, axis=1).columns
        )

        plt.figure(figsize=(vs, sh), dpi=dpi)
        sns.heatmap(df_comp[:n], annot=True)

    @decorator_elapsed_time
    def plot_pca(

        self, 
        variance=0.95, 
        svd_solver="auto"):

        scaler = StandardScaler()
        X_pca = scaler.fit_transform(self.X)

        pca = PCA(svd_solver=svd_solver)
        pca.fit(X_pca)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        dimension = np.argmax(cumsum >= variance) + 1

        print(f"Variance - {variance}, Dimension - {dimension}")

        explained_variance = []
        dimension = []

        for v in np.arange(0.01, 1.0, 0.05):
            d = np.argmax(cumsum >= v) + 1

            explained_variance.append(v)
            dimension.append(d)

        plt.plot(dimension, explained_variance)
        plt.xlabel("Number of Dimension")
        plt.ylabel("Variance Explained")
        plt.grid(alpha=0.2)
        plt.show()

    @decorator_elapsed_time
    def preprocessing(

        self, 
        mode="StandardScaler", 
        n_components=2):

        if mode == "MinMaxScaler":
            self.Scaler_mark = "on"
            self.PCA_mark = "off"
            scaler = MinMaxScaler()

        if mode == "StandardScaler":
            self.Scaler_mark = "on"
            self.PCA_mark = "off"
            scaler = StandardScaler()

        if mode in ["MinMaxScaler", "StandardScaler"]:
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_valid = scaler.transform(self.X_valid)
            self.X_test = scaler.transform(self.X_test)

            self.scaler = scaler

        if mode == "PCA":
            self.Scaler_mark = "on"
            self.PCA_mark = "on"

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

        return (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test)

    def data_split(

        self, 
        valid=0.2, 
        stratify=None):

        if stratify:
            stratify = self.df[stratify]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X, self.y, test_size=valid, stratify=stratify)

        return self.X_train, self.X_valid, self.y_train, self.y_valid

    def basemodel_common(

        self,
        estimator=None,
        mode=None,
        n_iter=10,
        params=None,
        cv=5,
        scoring=None,
        n_jobs=None,
        test_on="valid_data",
        n_points=1,):

        model = self.models_dic_base[estimator]

        if mode == "auto_grid":
            index_for_params_grid = {
                key.__class__.__name__: index
                for index, key in enumerate(list(self.models_dic_grid.keys()))}

            parameter_dic_grid = self.models_dic_grid[
                list(self.models_dic_grid.keys())[index_for_params_grid[estimator]]]

            search = GridSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                param_grid=parameter_dic_grid,
                scoring=scoring)

        if mode == "auto_random":
            index_for_params_random = {
                key.__class__.__name__: index
                for index, key in enumerate(list(self.models_dic_random.keys()))}

            parameter_dic_random = self.models_dic_random[
                list(self.models_dic_random.keys())[index_for_params_random[estimator]]]

            search = RandomizedSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                n_iter=n_iter,
                param_distributions=parameter_dic_random,
                scoring=scoring)

        if mode == "auto_bayes":
            index_for_params_random = {
                key.__class__.__name__: index
                for index, key in enumerate(list(self.models_dic_random.keys()))}

            parameter_dic_random = self.models_dic_random[
                list(self.models_dic_random.keys())[index_for_params_random[estimator]]]

            for key, value in parameter_dic_random.items():
                parameter_dic_random[key] = tuple(value)

            search = BayesSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                n_iter=n_iter,
                search_spaces=parameter_dic_random,
                scoring=scoring,
                n_points=n_points,
                verbose=0)

        if mode == "set_manual":
            params_with_brackets = {key: [value] for key, value in params.items()}

            search = GridSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                param_grid={**params_with_brackets},
                scoring=scoring,)

        if mode == "set_random":
            search = RandomizedSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                n_iter=n_iter,
                param_distributions={**params},
                scoring=scoring)

        if mode == "set_grid":
            search = GridSearchCV(
                model, cv=cv, n_jobs=n_jobs, param_grid={**params}, scoring=scoring)

        if mode == "set_bayes":
            search = BayesSearchCV(
                model,
                cv=cv,
                n_jobs=n_jobs,
                n_iter=n_iter,
                search_spaces={**params},
                scoring=scoring,
                n_points=n_points,
                verbose=2)

        search.fit(self.X_train, self.y_train)
        self.basemodel_model = search
        self.estimator_from_dic["basemodel"] = self.basemodel_model.best_estimator_


class Timeseries:

    def naive_forecast(self):

        naive_forecast_ = np.append(self.y.mean(), self.y[:-1])

        mean_absolute_error_valid = round(mean_absolute_error(self.y, naive_forecast_), 2)
        mean_squared_error_valid = round(np.sqrt(mean_squared_error(self.y, naive_forecast_)), 2)
        r2_score_valid = round(r2_score(self.y, naive_forecast_), 2)
        median_absolute_error_valid = round(median_absolute_error(self.y,naive_forecast_),2)
        mape_valid = np.abs((naive_forecast_ - self.y) / self.y).mean()
        variance_valid = round(np.var(naive_forecast_), 2)

        result_test_df = pd.DataFrame({"Naive_forecast": [
                    mean_absolute_error_valid,
                    mean_squared_error_valid,
                    r2_score_valid,
                    median_absolute_error_valid,
                    mape_valid,
                    variance_valid]},
            index=[
                "mean_absolute_error_valid",
                "mean_squared_error_valid",
                "r2_score_valid",
                "median_absolute_error_valid",
                "mape_valid",
                "variance_valid"])

        return result_test_df.transpose()

    def data_windows(

        self, 
        windows=30, 
        horizon=1, 
        valid=0.2, 
        test=0.1, 
        shuffle=True):

        self.data_windows_multy_mark = "off"

        data_array = self.y.values

        window_step = np.arange(windows + horizon)
        window_indexes = (
            window_step
            + np.arange(len(data_array) - (windows + horizon - 1))[:, np.newaxis])
        windowed_array = np.array([data_array[idx] for idx in window_indexes])

        if shuffle:
            num_samples = windowed_array.shape[0]
            shuffled_indices = np.random.permutation(num_samples)

            X = windowed_array[shuffled_indices, :-horizon]
            y = windowed_array[shuffled_indices, -horizon:]

        else:
            X = windowed_array[:, :-horizon]
            y = windowed_array[:, -horizon:]

        total_size = len(X)
        train_size = int(total_size * (1 - valid - test))
        valid_size = int(total_size * valid)

        self.X_train, self.X_valid, self.X_test = (
            X[:train_size],
            X[train_size : train_size + valid_size],
            X[train_size + valid_size :],
        )

        self.y_train, self.y_valid, self.y_test = (
            y[:train_size],
            y[train_size : train_size + valid_size],
            y[train_size + valid_size :],
        )

        self.train_size = train_size

    def plot_valid_vs_predicted(self, estimator_from="basemodel", estimator=None):
        if estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        predicted = []

        for i in self.X_valid:
            future_pred = model.predict(i.reshape(1, -1))
            predicted.append(future_pred)

        index = range(0, len(self.y_valid))

        plt.plot(index, self.y_valid, color="blue", label="Original Data")
        plt.plot(index, predicted, color="red", label="Future Forecast")

    def make_future_forecast(
        self, values, model, into_future, plot=False, regplot=True, moving_average=True
    ):
        if self.data_multy_windows_mark == "on":
            print("No possible to use with data_multy_windows")

        else:
            values.reset_index(inplace=True, drop=True)

            window_size = model.n_features_in_
            future_forecast_values = []
            future_forecast_index = []

            try:
                last_window = values[-window_size:].to_numpy()
            except:
                last_window = values[-window_size:]

            for i in range(into_future):
                future_pred = model.predict(last_window.reshape(1, -1))

                future_forecast_values.append(round(future_pred[0], 2))
                future_forecast_index.append(values.index[-1] + i + 1)
                last_window = np.append(last_window[1:], future_pred[0])

            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    values.index, values.values, color="blue", label="Original Data"
                )
                plt.plot(
                    future_forecast_index,
                    future_forecast_values,
                    color="red",
                    label="Future Forecast",
                )

                if regplot:
                    sns.regplot(
                        x=values.index,
                        y=values.values,
                        color="green",
                        scatter_kws={"s": 20},
                        label="Regression Line",
                    )

                if moving_average:
                    moving_average_forecast = []
                    for time in range(len(values.values) - window_size):
                        moving_average_forecast.append(
                            values.values[time : time + window_size].mean()
                        )
                    plt.plot(
                        values.index[window_size:],
                        moving_average_forecast,
                        color="yellow",
                        label="Moving Average",
                    )

                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.legend()
                plt.show()

            return future_forecast_values

    def data_windows_multy(self, windows=7, valid=0.2, test=0.1, shuffle=True):
        self.data_windows_multy_mark = "on"

        df = self.df.copy()

        for i in range(windows):
            df[f"{self.target}+{i + 1}"] = df[f"{self.target}"].shift(periods=i + 1)

        df = df.iloc[windows:]

        if shuffle:
            df = df.sample(frac=1)

        X = df.drop(f"{self.target}", axis=1)
        y = df[f"{self.target}"]

        total_size = len(X)
        train_size = int(total_size * (1 - valid - test))
        valid_size = int(total_size * valid)

        self.X_train, self.X_valid, self.X_test = (
            X[:train_size],
            X[train_size : train_size + valid_size],
            X[train_size + valid_size :],
        )

        self.y_train, self.y_valid, self.y_test = (
            y[:train_size],
            y[train_size : train_size + valid_size],
            y[train_size + valid_size :],
        )

    def make_future_forecast_multy(self, values, model, window_size):
        df = values[-window_size - 1 :].copy()

        for i in range(window_size):
            df[f"{self.target}+{i + 1}"] = df[f"{self.target}"].shift(periods=i + 1)

        df = df.iloc[window_size:]
        X = df.drop(f"{self.target}", axis=1)

        return model.predict(X)

    @staticmethod
    def mean_value_for_group(df, group_size=5, index_is_datetime=False, time="W-Mon"):
        if index_is_datetime:
            df_weekly = df.resample(time).mean()

            return df_weekly

        df.reset_index(inplace=True, drop=True)
        df = df.groupby(df.index // group_size).mean()

        return df

    @staticmethod
    def pivot_point(series, ave_size=1, min_n=None, max_n=None, plot=True):
        series.reset_index(inplace=True, drop=True)

        grouped_df = series.groupby(series.index // ave_size).mean()

        data_int = grouped_df.astype(int)
        data_int_sorted = data_int.sort_values()
        result_dict = dict(data_int_sorted.value_counts())

        if min_n:
            result_dict = {
                key: values
                for key, values in result_dict.items()
                if min_n <= key <= max_n
            }

        if plot:
            keys = list(result_dict.keys())
            values = list(result_dict.values())
            plt.bar(keys, values)
            plt.xlabel("Target")
            plt.ylabel("Mode")
            plt.title("Frequency")
            plt.show()

        return result_dict

    def windows_check(self, min_n=None, max_n=None, step=1, set_window=None):
        if min_n:
            window_sizes = range(min_n, max_n, step)

            mean_values_by_window = {}

            for window_size in window_sizes:
                windows = [
                    self.y[i : i + window_size]
                    for i in range(0, len(self.y) - window_size + 1)
                ]
                mean_values_by_window[window_size] = [
                    window.std() for window in windows
                ]

            variance_by_window = {}

            for window_size in window_sizes:
                mean_values = mean_values_by_window[window_size]
                variance_by_window[window_size] = pd.Series(mean_values).mean()

            result = pd.DataFrame(variance_by_window, index=["std"]).transpose()
            result.plot(kind="bar")

            self.windows_check_ = mean_values_by_window

        if set_window:
            plt.plot(
                range(0, len(self.y.values)),
                self.y.values,
                color="blue",
                label="Original Data",
            )

            for i in range(0, len(self.y.values) - set_window + 1, set_window):
                last_value = self.y.values[i + set_window]
                plt.axvline(x=i + set_window, color="red", linestyle="--", alpha=0.7)

            plt.show()


class Classifier(ModelTools):
    def decorator_elapsed_time(func):
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            elapsed_time = round(timeit.default_timer() - start_time, 2)
            print("Elapsed time:", elapsed_time)
            return result

        return wrapper

    def __init__(
        self, df, target, class_weight=None, probability=True, test=0.1, stratify=None
    ):
        df.index.name = "index"

        self.target = target

        X = df.drop({self.target}, axis=1)
        y = df[self.target]

        if stratify:
            stratify = df[stratify]

        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X, y, test_size=test, stratify=stratify
        )

        columns_to_keep = df.columns.drop(self.target)
        self.df = pd.DataFrame(data=self.X, columns=columns_to_keep)
        self.df[self.target] = self.y

        self.PCA_mark = "off"
        self.Scaler_mark = "off"
        self.TfidfVectorizer_mark = "off"

        self.estimator_from_dic = {}

        self.models_dic_base = {
            "DecisionTreeClassifier": DecisionTreeClassifier(class_weight=class_weight),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                class_weight=class_weight, warm_start=True
            ),
            "RandomForestClassifier": RandomForestClassifier(
                class_weight=class_weight, warm_start=True
            ),
            "GradientBoostingClassifier": GradientBoostingClassifier(warm_start=True),
            "XGBClassifier": XGBClassifier(),
            "SVC": SVC(probability=probability),
            "LogisticRegression": LogisticRegression(
                class_weight=class_weight, max_iter=200, warm_start=True
            ),
            "KNeighborsClassifier": KNeighborsClassifier(),
        }

        tree_dic_params = {
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2"],
        }

        tree_dic_params_reg = {
            "min_weight_fraction_leaf": [0.0, 0.1],
            "min_impurity_decrease": [0.0, 0.1],
            "ccp_alpha": [0.0, 0.1],
            "max_depth": [3, 10, 100],
        }

        self.models_dic_grid = {
            DecisionTreeClassifier(class_weight=class_weight): {
                "splitter": ["best", "random"]
            }
            | tree_dic_params,
            ExtraTreesClassifier(class_weight=class_weight, warm_start=True): {
                "n_estimators": [64, 100, 128]
            }
            | tree_dic_params,
            RandomForestClassifier(class_weight=class_weight, warm_start=True): {
                "n_estimators": [64, 100, 128]
            }
            | tree_dic_params,
            GradientBoostingClassifier(warm_start=True): {
                "n_estimators": [64, 100, 128]
            }
            | tree_dic_params,
            XGBClassifier(): {
                "eta": np.logspace(np.log10(0.1), np.log10(3.0), num=3),
                "min_split_loss": [0.1, 0.2, 0.3],
                "alpha": [0.01, 0.1, 1],
                "lambda": [0.01, 0.1, 1],
                "min_child_weight": [1, 5, 10],
            }
            | tree_dic_params,
            SVC(probability=probability): {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
            LogisticRegression(
                class_weight=class_weight,
                max_iter=200,
                penalty="elasticnet",
                warm_start=True,
            ): {
                "C": np.logspace(np.log10(0.01), np.log10(100.0), num=4),
                "solver": ["lbfgs", "sag", "saga"],
            },
            KNeighborsClassifier(): {
                "n_neighbors": [1, 2, 4, 8, 16, 32, 64],
                "weights": ["uniform", "distance"],
            },
        }

        self.models_dic_random = {
            DecisionTreeClassifier(class_weight=class_weight): {
                "splitter": ["best", "random"]
            }
            | tree_dic_params
            | tree_dic_params_reg,
            ExtraTreesClassifier(class_weight=class_weight, warm_start=True): {
                "n_estimators": range(64, 128)
            }
            | tree_dic_params
            | tree_dic_params_reg,
            RandomForestClassifier(class_weight=class_weight, warm_start=True): {
                "n_estimators": range(64, 128)
            }
            | tree_dic_params
            | tree_dic_params_reg,
            GradientBoostingClassifier(warm_start=True): {
                "n_estimators": range(64, 128)
            }
            | tree_dic_params
            | tree_dic_params_reg,
            XGBClassifier(): {
                "eta": np.logspace(np.log10(0.1), np.log10(3.0), num=10),
                "min_split_loss": [0.1, 0.2, 0.3],
                "alpha": [0.01, 0.1, 1],
                "lambda": [0.01, 0.1, 1],
                "min_child_weight": [1, 5, 10],
            }
            | tree_dic_params
            | tree_dic_params_reg,
            SVC(probability=probability): {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
            },
            LogisticRegression(
                class_weight=class_weight, max_iter=200, warm_start=True
            ): {
                "penalty": ["l1", "l2", "elasticnet"],
                "C": np.logspace(np.log10(0.01), np.log10(100.0), num=10),
                "solver": ["lbfgs", "liblinear", "sag", "saga"],
                "fit_intercept": [True, False],
            },
            KNeighborsClassifier(): {
                "n_neighbors": range(1, 64, 1),
                "weights": ["uniform", "distance"],
            },
        }

    @decorator_elapsed_time
    def tfidf(self):
        self.Scaler_mark = "on"
        self.TfidfVectorizer_mark = "on"
        self.PCA_mark = "off"
        scaler = TfidfVectorizer()

        self.X_train = self.X_train.squeeze()
        self.X_valid = self.X_valid.squeeze()
        self.X_test = self.X_test.squeeze()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_valid = scaler.transform(self.X_valid)
        self.X_test = scaler.transform(self.X_test)

        self.scaler = scaler

        return (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        )

    @decorator_elapsed_time
    def preanalyze(self, bins=20, alpha=0.5):
        # PCA plot
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_train)

        X_train_pca = pd.DataFrame(principal_components)

        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            x=X_train_pca[0],
            y=X_train_pca[1],
            data=pd.DataFrame(self.X),
            hue=self.y,
            alpha=alpha,
        )
        plt.xlabel("First principal component")
        plt.ylabel("Second Principal Component")

        # Distribution plot
        if len(pd.DataFrame(self.df)[f"{self.target}"].value_counts()) < 10:
            cluster_counts = pd.DataFrame(self.df)[f"{self.target}"].value_counts()
            plt.figure(figsize=(12, 4), dpi=200)
            plt.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%")
            plt.title(f"{self.target} Distribution")
            plt.xlabel(f"{self.target}")
            plt.ylabel("Count")
        else:
            plt.figure(figsize=(12, 4), dpi=200)
            sns.histplot(
                data=pd.DataFrame(self.df),
                x=f"{self.target}",
                kde=True,
                color="green",
                bins=bins,
            )
            plt.title(f"{self.target} Distribution")
            plt.xlabel(f"{self.target}")
            plt.ylabel("Count")

        plt.show()

        # Correlation plot
        corr_df = pd.DataFrame(self.df).corr()
        plt.figure(figsize=(12, 4), dpi=200)
        sns.barplot(
            x=corr_df[self.target].sort_values().iloc[1:-1].index,
            y=corr_df[self.target].sort_values().iloc[1:-1].values,
        )
        plt.title(f"Feature Correlation to {self.target}")
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        plt.xticks(rotation=90)

        plt.show()

    @decorator_elapsed_time
    def ensemble(
        self,
        estimators=None,
        tuner="default",
        cv=5,
        scoring="accuracy",
        n_iter=10,
        n_jobs=2,
        average="macro",
    ):
        self.result_df_ensamble = pd.DataFrame()
        self.ensemble_models = {}

        if tuner == "GridSearchCV":
            model_dict_for_tuner = self.models_dic_grid
        else:
            model_dict_for_tuner = self.models_dic_random

        if estimators == "tree_only":
            estimators = list(self.models_dic_base.keys())[:5]
        if estimators == "no_tree":
            estimators = list(self.models_dic_base.keys())[5:]

        if estimators:
            model_dict_for_tuner = {
                key: model_dict_for_tuner[key]
                for key, value in model_dict_for_tuner.items()
                if key.__class__.__name__ in estimators
            }

        for key, value in tqdm(
            model_dict_for_tuner.items(), desc="Tuning Ensemble Models"
        ):
            ensemble_model_name = key.__class__.__name__
            print(f"Model: {ensemble_model_name}")

            if tuner == "GridSearchCV":
                ensemble_search = GridSearchCV(
                    key, cv=cv, n_jobs=n_jobs, param_grid=value, scoring=scoring
                )

            if tuner == "RandomizedSearchCV":
                ensemble_search = RandomizedSearchCV(
                    key,
                    cv=cv,
                    n_jobs=n_jobs,
                    param_distributions=value,
                    scoring=scoring,
                    n_iter=n_iter,
                )

            if tuner == "default":
                ensemble_search = GridSearchCV(
                    key, cv=cv, n_jobs=n_jobs, param_grid={}, scoring=scoring
                )

            ensemble_search.fit(self.X_train, self.y_train)
            self.ensemble_models[ensemble_model_name] = ensemble_search

            df_iter_model_ensemble = self.result_test_df(ensemble_search)
            df_iter_model_ensemble = df_iter_model_ensemble.transpose()
            df_iter_model_ensemble.rename(
                columns={df_iter_model_ensemble.columns[-1]: str(key)}, inplace=True
            )
            self.result_df_ensamble = pd.concat(
                [self.result_df_ensamble, df_iter_model_ensemble], axis=1
            )

        self.result_df_ensamble = self.result_df_ensamble.transpose()
        self.result_df_ensamble.index = [
            index[: index.find("(")] for index in self.result_df_ensamble.index
        ]

        fig, axes = plt.subplots()
        sns.scatterplot(
            x=self.result_df_ensamble["precision_valid"],
            y=self.result_df_ensamble["recall_valid"],
            hue=self.result_df_ensamble.index,
            size=self.result_df_ensamble["f1_valid"],
            sizes=(50, 200),
            ax=axes,
        )
        axes.set_xlabel("Precision")
        axes.set_ylabel("Recall")
        axes.legend(loc=(1.1, 0.0))

        self.estimator_from_dic["ensemble"] = {
            name: model.best_estimator_ for name, model in self.ensemble_models.items()
        }

        return self.result_df_ensamble.iloc[:, :7]

    @decorator_elapsed_time
    def voting(self, voting="soft", test_on="valid_data", average="macro"):
        self.voting_model = VotingClassifier(
            estimators=[
                (key, value.best_estimator_)
                if hasattr(value, "best_estimator_")
                else (key, value)
                for key, value in self.ensemble_models.items()
            ],
            voting=voting,
        )

        self.voting_model.fit(self.X_train, self.y_train)
        self.estimator_from_dic["voting"] = self.voting_model

        if test_on == "valid_data":
            return self.result_test_df(self.voting_model, average).iloc[:, :7]
        if test_on == "test_data":
            return self.result_test_df(self.voting_model, average)

    @decorator_elapsed_time
    def basemodel(
        self,
        estimator="LogisticRegression",
        mode="auto_random",
        n_iter=10,
        params=None,
        cv=5,
        scoring="accuracy",
        n_jobs=None,
        test_on="valid_data",
        average="macro",
        n_points=1,
    ):
        self.basemodel_common(
            estimator=estimator,
            mode=mode,
            n_iter=n_iter,
            params=params,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            test_on=test_on,
            n_points=n_points,
        )

        if test_on == "valid_data":
            return self.result_test_df(search, average).iloc[:, :7]
        if test_on == "test_data":
            return self.result_test_df(search, average)

    @decorator_elapsed_time
    def tuning(
        self,
        estimator_from="default",
        estimator="KNeighborsClassifier",
        target="n_neighbors",
        min_n=1,
        max_n=30,
        step=1,
        set_target=None,
        test_on="valid_data",
        average="macro",
        params=None,
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if estimator_from != "default":
            params = model.get_params()

        if params and target in params:
            del params[target]

        if set_target:
            params_dict = {target: set_target}
            if params:
                params_dict.update(params)
            model.set_params(**params_dict)
            model.fit(self.X_train, self.y_train)

            self.tuning_model = model
            self.estimator_from_dic["tuning"] = self.tuning_model

            if test_on == "valid_data":
                return self.result_test_df(self.tuning_model, average).iloc[:, :7]
            if test_on == "test_data":
                return self.result_test_df(self.tuning_model, average)

        else:
            metrics = ["precision", "recall", "f1", "accuracy"]
            test_error_rates = {metric: [] for metric in metrics}

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.flatten()

            for i in tqdm(
                np.arange(min_n, max_n, step), desc=f"Checking model for {target}"
            ):
                try:
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    model.set_params(**params_dict)
                except InvalidParameterError:
                    i = i.astype("int")
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    model.set_params(**params_dict)

                model.fit(self.X_train, self.y_train)
                y_pred_valid = model.predict(self.X_valid)

                test_error_rates["accuracy"].append(
                    accuracy_score(self.y_valid, y_pred_valid)
                )
                test_error_rates["precision"].append(
                    precision_score(self.y_valid, y_pred_valid, average=average)
                )
                test_error_rates["recall"].append(
                    recall_score(self.y_valid, y_pred_valid, average=average)
                )
                test_error_rates["f1"].append(
                    f1_score(self.y_valid, y_pred_valid, average=average)
                )

            axs[0].plot(np.arange(min_n, max_n, step), test_error_rates["precision"])
            axs[0].set_ylabel("precision")
            axs[0].set_xlabel(f"{target}")
            axs[0].grid(True)

            axs[1].plot(np.arange(min_n, max_n, step), test_error_rates["recall"])
            axs[1].set_ylabel("recall")
            axs[1].set_xlabel(f"{target}")
            axs[1].grid(True)

            axs[2].plot(np.arange(min_n, max_n, step), test_error_rates["f1"])
            axs[2].set_ylabel("f1")
            axs[2].set_xlabel(f"{target}")
            axs[2].grid(True)

            axs[3].plot(np.arange(min_n, max_n, step), test_error_rates["accuracy"])
            axs[3].set_ylabel("accuracy")
            axs[3].set_xlabel(f"{target}")
            axs[3].grid(True)

            plt.tight_layout()
            plt.show()

    @decorator_elapsed_time
    def ada(
        self,
        estimator_from="default",
        estimator="DecisionTreeClassifier",
        n_estimators=50,
        learning_rate=1.0,
        test_on="valid_data",
        average="macro",
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        try:
            self.ada_model = AdaBoostClassifier(
                base_estimator=model,
                algorithm="SAMME.R",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
            )

            self.ada_model.fit(self.X_train, self.y_train)

        except:
            self.ada_model = AdaBoostClassifier(
                base_estimator=model,
                algorithm="SAMME",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
            )

            self.ada_model.fit(self.X_train, self.y_train)

        self.estimator_from_dic["ada"] = self.ada_model

        if test_on == "valid_data":
            return self.result_test_df(self.ada_model, average).iloc[:, :7]
        if test_on == "test_data":
            return self.result_test_df(self.ada_model, average)

    @decorator_elapsed_time
    def bagging(
        self,
        estimator_from="default",
        estimator="DecisionTreeClassifier",
        n_estimators=500,
        max_samples=0.1,
        bootstrap=True,
        n_jobs=1,
        oob_score=True,
        max_features=1.0,
        bootstrap_features=True,
        test_on="valid_data",
        average="macro",
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        try:
            self.bagging_model = BaggingClassifier(
                model,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap,
                n_jobs=n_jobs,
                oob_score=oob_score,
                max_features=max_features,
                bootstrap_features=bootstrap_features,
                warm_start=True,
            )

            self.bagging_model.fit(self.X_train, self.y_train)

            print(f"oob_score - {self.bagging_model.oob_score_}")

        except ValueError:
            self.bagging_model = BaggingClassifier(
                model,
                n_estimators=n_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap,
                n_jobs=n_jobs,
                max_features=max_features,
                bootstrap_features=bootstrap_features,
                warm_start=True,
            )

            self.bagging_model.fit(self.X_train, self.y_train)

        self.estimator_from_dic["bagging"] = self.bagging_model

        if test_on == "valid_data":
            return self.result_test_df(self.bagging_model, average).iloc[:, :7]
        if test_on == "test_data":
            return self.result_test_df(self.bagging_model, average)

    @decorator_elapsed_time
    def threshold(
        self,
        estimator_from="default",
        estimator="LogisticRegression",
        set_threshold=None,
        test_on="valid_data",
        average="macro",
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if estimator_from == "fine_tuning":
            X = self.X_test
            y = self.y_test
        else:
            X = self.X_valid
            y = self.y_valid

        class ThresholdClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self, threshold=0.5, estimator=None):
                self.threshold = threshold
                self.estimator = estimator

            def fit(self, X, y):
                self.estimator.fit(X, y)
                return self

            def predict(self, X):
                return (self.estimator.predict_proba(X)[:, 1] >= self.threshold).astype(
                    int
                )

            def predict_proba(self, X):
                return self.estimator.predict_proba(X)

            def get_params(self, deep=True):
                return {"threshold": self.threshold, "estimator": self.estimator}

            def set_params(self, **parameters):
                for parameter, value in parameters.items():
                    setattr(self, parameter, value)
                return self

            def classes_(self):
                return self.estimator.classes_

            def __len__(self):
                return len(self.estimator.classes_)

        def threshold_function(threshold, estimator):
            return ThresholdClassifier(threshold, estimator)

        if set_threshold:
            self.threshold_model = make_pipeline(
                threshold_function(set_threshold, model)
            )
            self.estimator_from_dic["threshold"] = self.threshold_model

            if test_on == "valid_data":
                return self.result_test_df(self.threshold_model, average).iloc[:, :7]
            if test_on == "test_data":
                return self.result_test_df(self.threshold_model, average)

        else:
            metrics = ["precision", "recall", "f1", "accuracy"]

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.flatten()

            for idx, metric in enumerate(metrics):
                test_error_rates = []

                for i in tqdm(
                    np.arange(0.01, 0.99, 0.01),
                    desc=f"Threshold assessment for {metric}",
                ):
                    threshold_model = make_pipeline(threshold_function(i, model))
                    y_pred_valid = threshold_model.predict(X)

                    if metric == "accuracy":
                        metric_score = accuracy_score(y, y_pred_valid)
                    elif metric == "precision":
                        metric_score = precision_score(y, y_pred_valid, average=average)
                    elif metric == "recall":
                        metric_score = recall_score(y, y_pred_valid, average=average)
                    elif metric == "f1":
                        metric_score = f1_score(y, y_pred_valid, average=average)

                    test_error_rates.append(metric_score)

                axs[idx].plot(
                    np.arange(0.01, 0.99, 0.01),
                    test_error_rates,
                    label=f"{metric} / threshold ratio",
                )
                axs[idx].legend()
                axs[idx].set_ylabel(f"{metric}")
                axs[idx].set_xlabel(f"threshold")
                axs[idx].grid(True)

            plt.tight_layout()
            plt.show()

    def result_test_df(self, model, average="macro"):
        y_pred_valid = model.predict(self.X_valid)
        accuracy_valid = round(accuracy_score(self.y_valid, y_pred_valid), 2)
        precision_valid = round(
            precision_score(self.y_valid, y_pred_valid, average=average), 2
        )
        recall_valid = round(
            recall_score(self.y_valid, y_pred_valid, average=average), 2
        )
        f1_valid = round(f1_score(self.y_valid, y_pred_valid, average=average), 2)

        y_pred_test = model.predict(self.X_test)
        accuracy_test = round(accuracy_score(self.y_test, y_pred_test), 2)
        precision_test = round(
            precision_score(self.y_test, y_pred_test, average=average), 2
        )
        recall_test = round(recall_score(self.y_test, y_pred_test, average=average), 2)
        f1_test = round(f1_score(self.y_test, y_pred_test, average=average), 2)

        if self.TfidfVectorizer_mark == "on":
            variance_valid = "n/a"
            variance_test = "n/a"
        else:
            variance_valid = round(np.var(y_pred_valid), 2)
            variance_test = round(np.var(y_pred_test), 2)

        try:
            parameters = model.best_estimator_.get_params()
            building_time = model.cv_results_["mean_fit_time"].sum()

        except AttributeError:
            parameters = model.get_params()
            building_time = "|"

        result_test_df = pd.DataFrame(
            {
                f"{model.__class__.__name__}": [
                    parameters,
                    building_time,
                    accuracy_valid,
                    precision_valid,
                    recall_valid,
                    f1_valid,
                    variance_valid,
                    accuracy_test,
                    precision_test,
                    recall_test,
                    f1_test,
                    variance_test,
                ]
            },
            index=[
                "parameters",
                "building_time",
                "accuracy_valid",
                "precision_valid",
                "recall_valid",
                "f1_valid",
                "variance_valid",
                "accuracy_test",
                "precision_test",
                "recall_test",
                "f1_test",
                "variance_test",
            ],
        )

        result_test_df = result_test_df.transpose()

        return result_test_df

    def get_pipe(self, estimator_from, estimator=None):
        if estimator:
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if self.Scaler_mark == "on":
            self.build_pipe = make_pipeline(self.scaler, model)
        if self.PCA_mark == "on":
            self.build_pipe = make_pipeline(self.scaler, self.pca, model)

        return self.build_pipe

    def plot_tree(self, params={"max_depth": 3}, dpi=300, save=False):
        tree = DecisionTreeClassifier()
        tree.set_params(**params)
        tree.fit(self.X, self.y)

        self.plot_tree_df = pd.DataFrame(
            index=self.X.columns,
            data=np.round(tree.feature_importances_, 2),
            columns=["Feature Importance"],
        ).sort_values("Feature Importance", ascending=False)

        plt.figure(figsize=(12, 8), dpi=dpi)
        class_names = [str(cls) for cls in self.y.unique()]
        plot_tree(
            tree,
            filled=True,
            feature_names=self.X.columns,
            proportion=True,
            rounded=True,
            precision=2,
            class_names=class_names,
            label="root",
        )

        if save:
            plt.savefig("plot_tree.png")

        return self.plot_tree_df.transpose()

    def plot_mat(
        self,
        estimator_from="basemodel",
        estimator=None,
        test_on="valid_data",
        procent=True,
    ):
        np.seterr(divide="ignore")

        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        y_pred_valid = model.predict(self.X_valid)
        cm_valid = confusion_matrix(self.y_valid, y_pred_valid)
        y_pred_test = model.predict(self.X_test)
        cm_test = confusion_matrix(self.y_test, y_pred_test)

        if test_on == "valid_data":
            cm = cm_valid
            print(classification_report(self.y_valid, y_pred_valid))
        if test_on == "test_data":
            cm = cm_test
            print(classification_report(self.y_test, y_pred_test))

        if procent:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        classes = model.classes_

        try:
            ax = sns.heatmap(
                np.round(cm, 2),
                annot=True,
                xticklabels=classes,
                yticklabels=classes,
                cmap="Greens",
            )

        except TypeError:
            ax = sns.heatmap(np.round(cm, 2), annot=True, cmap="Greens")

        ax.set(xlabel="Predict", ylabel="Actual")

        if procent == False and test_on == "test_data":
            test_num = [
                pair[0] for pair in self.get_num_elements_per_class_and_recall(cm_test)
            ]
            valid_rec = [
                pair[1] for pair in self.get_num_elements_per_class_and_recall(cm_valid)
            ]
            data = list(zip(test_num, valid_rec))

            return self.binomial(data)

    @staticmethod
    def get_num_elements_per_class_and_recall(confusion_matrix):
        num_classes = confusion_matrix.shape[0]
        num_elements_list = []
        recall_list = []

        for class_index in range(num_classes):
            TP = confusion_matrix[class_index, class_index]
            FN = np.sum(confusion_matrix[class_index, :]) - TP
            num_elements = TP + FN
            recall = TP / (TP + FN)
            num_elements_list.append(num_elements)
            recall_list.append(recall)

        data = list(zip(num_elements_list, recall_list))

        return data

    def plot_precision_recall(
        self, estimator_from="basemodel", estimator=None, test_on="valid_data"
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if test_on == "valid_data":
            X = self.X_valid
            y = self.y_valid
        if test_on == "test_data":
            X = self.X_test
            y = self.y_test

        plot_precision_recall(y, model.predict_proba(X))

        plt.tight_layout()
        plt.show()

    def plot_roc(
        self, estimator_from="basemodel", estimator=None, test_on="valid_data"
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if test_on == "valid_data":
            X = self.X_valid
            y = self.y_valid
        if test_on == "test_data":
            X = self.X_test
            y = self.y_test

        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def binomial(total=None, recall=None):
        def binomial_binary(total, recall):
            probability = [binom.pmf(k, total, recall) for k in range(1, total + 1)]

            cumsum = np.cumsum(probability)
            num = range(1, total + 1)

            cumsum = list(map(lambda x: round(1 - x, 2), cumsum))
            my_dict = dict(zip(cumsum, num))

            plt.plot(list(my_dict.values()), list(my_dict.keys()))
            plt.xlabel("Number of Successes")
            plt.ylabel("Probability")
            plt.title("Binomial Distribution")
            plt.grid()
            plt.show()

            sorted_values = sorted(my_dict.values())
            percentile_90 = int(np.percentile(sorted_values, 75))
            percentile_10 = int(np.percentile(sorted_values, 25))

            probability_sum = sum(
                binom.pmf(k, total, recall)
                for k in range(percentile_10, percentile_90 + 1)
            )
            print(
                f"The probability of obtaining the number of successes between {percentile_10} and {percentile_90}, inclusive, is: {round(probability_sum, 2)}"
            )

            return my_dict

        def binomial_multy(data):
            results = []

            for total, recall in data:
                probability = [binom.pmf(k, total, recall) for k in range(1, total + 1)]
                cumsum = np.cumsum(probability)
                num = range(1, total + 1)
                cumsum = list(map(lambda x: round(1 - x, 2), cumsum))
                my_dict = dict(zip(cumsum, num))

                sorted_values = sorted(my_dict.values())
                percentile_10 = int(np.percentile(sorted_values, 25))
                percentile_90 = int(np.percentile(sorted_values, 75))

                probability_sum = sum(
                    binom.pmf(k, total, recall)
                    for k in range(percentile_10, percentile_90 + 1)
                )

                result_dict = {
                    "Total": total,
                    "Recall": recall,
                    "Percentile_25": percentile_10,
                    "Percentile_75": percentile_90,
                    "Probability": round(probability_sum, 2),
                }

                results.append(result_dict)

            return pd.DataFrame(results)

        if isinstance(total, list):
            return binomial_multy(total)
        else:
            return binomial_binary(total, recall)


class Regression(ModelTools, Timeseries):
    def decorator_elapsed_time(func):
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            elapsed_time = round(timeit.default_timer() - start_time, 2)
            print("Elapsed time:", elapsed_time)
            return result

        return wrapper

    def __init__(self, df, target=None, test=0.1, stratify=None, mode="normal"):
        df.index.name = "index"

        self.data_multy_windows_mark = "off"

        if mode == "timeseries":
            self.y = df

        else:
            self.target = target

            X = df.drop({self.target}, axis=1)
            y = df[self.target]

            if stratify:
                stratify = df[stratify]

            if mode == "normal":
                self.X, self.X_test, self.y, self.y_test = train_test_split(
                    X, y, test_size=test, stratify=stratify
                )

            if mode == "sequential":
                train_split_index = int(len(df.index) * (1 - test))

                self.X = X.loc[:train_split_index]
                self.X_test = X.loc[train_split_index:]
                self.y = y.loc[:train_split_index]
                self.y_test = y.loc[train_split_index:]

            if mode == "timeseries_multy":
                self.X = df.drop({self.target}, axis=1)
                self.y = df[self.target]

            columns_to_keep = df.columns.drop(self.target)
            self.df = pd.DataFrame(data=self.X, columns=columns_to_keep)
            self.df[self.target] = self.y

        self.PCA_mark = "off"
        self.Scaler_mark = "off"
        self.Poly_mark = "off"

        self.estimator_from_dic = {}

        self.models_dic_base = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(warm_start=True),
            "ElasticNet": ElasticNet(warm_start=True),
            "HuberRegressor": HuberRegressor(warm_start=True),
            "RandomForestRegressor": RandomForestRegressor(warm_start=True),
            "GradientBoostingRegressor": GradientBoostingRegressor(warm_start=True),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
        }

        tree_dic_params = {
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2"],
        }

        tree_dic_params_reg = {
            "min_weight_fraction_leaf": [0.0, 0.1],
            "min_impurity_decrease": [0.0, 0.1],
            "ccp_alpha": [0.0, 0.1],
            "max_depth": [3, 10, 100],
        }

        self.models_dic_grid = {
            LinearRegression(): {},
            Ridge(): {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0], "positive": [True, False]},
            Lasso(warm_start=True): {
                "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
                "positive": [True, False],
            },
            ElasticNet(warm_start=True): {
                "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "positive": [True, False],
            },
            HuberRegressor(warm_start=True): {
                "epsilon": [1.35, 1.5, 1.75],
                "alpha": [0.0001, 0.001, 0.01],
                "fit_intercept": [True, False],
            },
            RandomForestRegressor(warm_start=True): {"n_estimators": [64, 100, 128]}
            | tree_dic_params,
            GradientBoostingRegressor(warm_start=True): {"n_estimators": [64, 100, 128]}
            | tree_dic_params,
            SVR(): {
                "kernel": ["linear", "rbf", "poly"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "epsilon": [0.05, 0.1, 0.2],
            },
            KNeighborsRegressor(): {
                "n_neighbors": [1, 2, 4, 8, 16, 32, 64],
                "weights": ["uniform", "distance"],
            },
        }

        self.models_dic_random = {
            LinearRegression(): {},
            Ridge(): {"alpha": np.logspace(-3, 1, 10), "positive": [True, False]},
            Lasso(warm_start=True): {
                "alpha": np.logspace(-3, 1, 10),
                "positive": [True, False],
            },
            ElasticNet(warm_start=True): {
                "alpha": np.logspace(-3, 1, 100),
                "l1_ratio": np.linspace(0.1, 1.0, 100),
                "positive": [True, False],
            },
            HuberRegressor(warm_start=True): {
                "epsilon": np.linspace(1.25, 2.0, 10),
                "alpha": np.logspace(-4, -1, 10),
                "fit_intercept": [True, False],
            },
            RandomForestRegressor(warm_start=True): {"n_estimators": range(64, 128)}
            | tree_dic_params,
            GradientBoostingRegressor(warm_start=True): {"n_estimators": range(64, 128)}
            | tree_dic_params,
            SVR(): {
                "kernel": ["linear", "rbf", "poly"],
                "C": np.logspace(-3, 1, 10),
                "gamma": ["scale", "auto"],
                "epsilon": [0.05, 0.1, 0.2],
            },
            KNeighborsRegressor(): {
                "n_neighbors": range(1, 64, 1),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": list(range(1, 101, 1)),
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            },
        }

    @decorator_elapsed_time
    def poly(
        self,
        high_range=4,
        set_dergee=None,
        cv=2,
        scoring="neg_mean_squared_error",
        include_bias=False,
    ):
        if set_dergee:
            self.Poly_mark = "on"

            poly = PolynomialFeatures(degree=set_dergee, include_bias=include_bias)
            self.X_train = poly.fit_transform(self.X_train)
            self.X_valid = poly.transform(self.X_valid)
            self.X_test = poly.transform(self.X_test)

            self.poly = poly

            return (
                self.X_train,
                self.X_valid,
                self.X_test,
                self.y_train,
                self.y_valid,
                self.y_test,
            )

        else:
            train_rmse_errors = []
            test_rmse_errors = []
            cross_poly = []

            for i in range(1, high_range):
                polynomial_converter = PolynomialFeatures(
                    degree=i, include_bias=include_bias
                )
                poly_features = polynomial_converter.fit_transform(self.X)
                X_train, X_test, y_train, y_test = train_test_split(
                    poly_features, self.y, test_size=0.3
                )

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                model = LinearRegression(fit_intercept=True)
                model.fit(X_train, y_train)

                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_RMSE = np.sqrt(mean_squared_error(y_train, train_pred))
                test_RMSE = np.sqrt(mean_squared_error(y_test, test_pred))

                train_rmse_errors.append(train_RMSE)
                test_rmse_errors.append(test_RMSE)

                scores_poly = cross_val_score(
                    model, poly_features, self.y, cv=cv, scoring=scoring
                )
                cross_poly.append(np.sqrt(-scores_poly).mean())

            poly_df = pd.DataFrame(
                {
                    "train_rmse_errors": train_rmse_errors,
                    "test_rmse_errors": test_rmse_errors,
                    "cross": cross_poly,
                },
                index=range(1, len(train_rmse_errors) + 1),
            )

            plt.plot(
                range(1, high_range), train_rmse_errors[: high_range - 1], label="TRAIN"
            )
            plt.plot(
                range(1, high_range), test_rmse_errors[: high_range - 1], label="TEST"
            )
            plt.xlabel("Polynomial Complexity")
            plt.ylabel("RMSE")
            plt.legend()
            plt.show()

            return poly_df.transpose()

    @decorator_elapsed_time
    def preanalyze(self, alpha=0.5):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.X)

        pca_2 = PCA(n_components=2)
        principal_components_2 = pca_2.fit_transform(X_train)
        d2 = pd.DataFrame(principal_components_2)

        pca_1 = PCA(n_components=1)
        principal_components_1 = pca_1.fit_transform(X_train)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18), sharex=True)

        sns.regplot(data=self.df, x=principal_components_1, y=self.y, ax=ax1)
        ax1.set_ylabel("Target variable")
        ax1.set_title(
            f"Explained variance ratio: {np.sum(pca_1.explained_variance_ratio_):.2f}"
        )

        sns.scatterplot(x=d2[0], y=d2[1], data=self.df, hue=self.y, alpha=0.5, ax=ax2)
        ax2.set_ylabel("Second principal component")
        ax2.set_title(
            f"Explained variance ratio: {pca_2.explained_variance_ratio_.sum():.2f}"
        )

        corr_df = self.df.corr()
        target_corr = corr_df[self.df.columns[-1]].drop(self.df.columns[-1])
        sns.barplot(
            x=target_corr.sort_values().index,
            y=target_corr.sort_values().values,
            ax=ax3,
        )
        ax3.set_xlabel("Features")
        ax3.set_ylabel("Correlation")
        ax3.set_title(f"Feature Correlation to {self.df.columns[-1]}")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=60)

        plt.tight_layout()
        plt.show()

    @decorator_elapsed_time
    def ensemble(
        self,
        estimators=None,
        tuner="default",
        cv=5,
        scoring="neg_mean_absolute_error",
        n_iter=10,
        n_jobs=2,
    ):
        self.result_df_ensamble = pd.DataFrame()
        self.ensemble_models = {}

        if tuner == "GridSearchCV":
            model_dict_for_tuner = self.models_dic_grid
        if tuner in ["RandomizedSearchCV", "default"]:
            model_dict_for_tuner = self.models_dic_random

        if estimators == "linear_only":
            estimators = list(self.models_dic_base.keys())[:5]
        if estimators == "no_linear":
            estimators = list(self.models_dic_base.keys())[5:]

        if estimators:
            model_dict_for_tuner = {
                key: model_dict_for_tuner[key]
                for key, value in model_dict_for_tuner.items()
                if key.__class__.__name__ in estimators
            }

        for key, value in tqdm(
            model_dict_for_tuner.items(), desc="Tuning Ensemble Models"
        ):
            ensemble_model_name = key.__class__.__name__
            print(f"Model: {ensemble_model_name}")

            if tuner == "GridSearchCV":
                ensemble_search = GridSearchCV(
                    key, cv=cv, n_jobs=n_jobs, param_grid=value, scoring=scoring
                )

            if tuner == "RandomizedSearchCV":
                ensemble_search = RandomizedSearchCV(
                    key,
                    cv=cv,
                    n_jobs=n_jobs,
                    param_distributions=value,
                    scoring=scoring,
                    n_iter=n_iter,
                )

            if tuner == "default":
                ensemble_search = GridSearchCV(
                    key, cv=cv, n_jobs=n_jobs, param_grid={}, scoring=scoring
                )

            ensemble_search.fit(self.X_train, self.y_train)
            self.ensemble_models[ensemble_model_name] = ensemble_search

            df_iter_model_ensemble = self.result_test_df(ensemble_search)
            df_iter_model_ensemble = df_iter_model_ensemble.transpose()
            df_iter_model_ensemble.rename(
                columns={df_iter_model_ensemble.columns[-1]: str(key)}, inplace=True
            )
            self.result_df_ensamble = pd.concat(
                [self.result_df_ensamble, df_iter_model_ensemble], axis=1
            )

        self.result_df_ensamble = self.result_df_ensamble.transpose()
        self.result_df_ensamble.index = [
            index[: index.find("(")] for index in self.result_df_ensamble.index
        ]

        fig, axes = plt.subplots()
        sns.scatterplot(
            x=self.result_df_ensamble["mean_absolute_error_valid"],
            y=self.result_df_ensamble["mean_squared_error_valid"],
            hue=self.result_df_ensamble.index,
            size=self.result_df_ensamble["r2_score_valid"],
            sizes=(50, 200),
            ax=axes,
        )
        axes.set_xlabel("mean_absolute_error")
        axes.set_ylabel("mean_squared_error")
        axes.legend(loc=(1.1, 0.0))

        self.estimator_from_dic["ensemble"] = {
            name: model.best_estimator_ for name, model in self.ensemble_models.items()
        }

        return self.result_df_ensamble.iloc[:, :8]

    @decorator_elapsed_time
    def voting(self, test_on="valid_data"):
        self.voting_model = VotingRegressor(
            estimators=[
                (key, value.best_estimator_)
                if hasattr(value, "best_estimator_")
                else (key, value)
                for key, value in self.ensemble_models.items()
            ]
        )

        self.voting_model.fit(self.X_train, self.y_train)
        self.estimator_from_dic["voting"] = self.voting_model

        if test_on == "valid_data":
            return self.result_test_df(self.voting_model).iloc[:, :8]
        if test_on == "test_data":
            return self.result_test_df(self.voting_model)

    @decorator_elapsed_time
    def basemodel(
        self,
        estimator="ElasticNet",
        mode="auto_random",
        n_iter=10,
        params=None,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=None,
        test_on="valid_data",
        n_points=1,
    ):
        self.basemodel_common(
            estimator=estimator,
            mode=mode,
            n_iter=n_iter,
            params=params,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            test_on=test_on,
            n_points=n_points,
        )

        if test_on == "valid_data":
            return self.result_test_df(self.basemodel_model).iloc[:, :8]
        if test_on == "test_data":
            return self.result_test_df(self.basemodel_model)

    @decorator_elapsed_time
    def tuning(
        self,
        estimator_from="default",
        estimator="ElasticNet",
        target="l1_ratio",
        min_n=0.1,
        max_n=1.0,
        step=0.1,
        set_target=None,
        test_on="valid_data",
        params=None,
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if estimator_from != "default":
            params = model.get_params()

        if params and target in params:
            del params[target]

        if set_target:
            params_dict = {target: set_target}
            if params:
                params_dict.update(params)
            model.set_params(**params_dict)
            model.fit(self.X_train, self.y_train)

            self.tuning_model = model
            self.estimator_from_dic["tuning"] = self.tuning_model

            if test_on == "valid_data":
                return self.result_test_df(self.tuning_model).iloc[:, :8]
            if test_on == "test_data":
                return self.result_test_df(self.tuning_model)

        else:
            metrics = [
                "mean_absolute_error",
                "mean_squared_error",
                "r2_score",
                "median_absolute_error",
            ]
            test_error_rates = {metric: [] for metric in metrics}

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.flatten()

            for i in tqdm(
                np.arange(min_n, max_n, step), desc=f"Checking model for {target}"
            ):
                try:
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    model.set_params(**params_dict)

                except InvalidParameterError:
                    i = i.astype("int")
                    params_dict = {target: i}
                    if params:
                        params_dict.update(params)
                    model.set_params(**params_dict)

                model.fit(self.X_train, self.y_train)
                y_pred_valid = model.predict(self.X_valid)

                test_error_rates["mean_absolute_error"].append(
                    mean_absolute_error(self.y_valid, y_pred_valid)
                )
                test_error_rates["mean_squared_error"].append(
                    mean_squared_error(self.y_valid, y_pred_valid)
                )
                test_error_rates["r2_score"].append(
                    r2_score(self.y_valid, y_pred_valid)
                )
                test_error_rates["median_absolute_error"].append(
                    median_absolute_error(self.y_valid, y_pred_valid)
                )

            axs[0].plot(
                np.arange(min_n, max_n, step), test_error_rates["mean_absolute_error"]
            )
            axs[0].set_ylabel("mean_absolute_error")
            axs[0].set_xlabel(f"{target}")
            axs[0].grid(True)

            axs[1].plot(
                np.arange(min_n, max_n, step), test_error_rates["mean_squared_error"]
            )
            axs[1].set_ylabel("mean_squared_error")
            axs[1].set_xlabel(f"{target}")
            axs[1].grid(True)

            axs[2].plot(np.arange(min_n, max_n, step), test_error_rates["r2_score"])
            axs[2].set_ylabel("r2_score")
            axs[2].set_xlabel(f"{target}")
            axs[2].grid(True)

            axs[3].plot(
                np.arange(min_n, max_n, step), test_error_rates["median_absolute_error"]
            )
            axs[3].set_ylabel("median_absolute_error")
            axs[3].set_xlabel(f"{target}")
            axs[3].grid(True)

            plt.tight_layout()
            plt.show()

    @decorator_elapsed_time
    def ada(
        self,
        estimator_from="default",
        estimator="ElasticNet",
        n_estimators=50,
        learning_rate=1.0,
        test_on="valid_data",
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        self.ada_model = AdaBoostRegressor(
            base_estimator=model, n_estimators=n_estimators, learning_rate=learning_rate
        )

        self.ada_model.fit(self.X_train, self.y_train)
        self.estimator_from_dic["ada"] = self.ada_model

        if test_on == "valid_data":
            return self.result_test_df(self.ada_model).iloc[:, :8]
        if test_on == "test_data":
            return self.result_test_df(self.ada_model)

    @decorator_elapsed_time
    def bagging(
        self,
        estimator_from="default",
        estimator="ElasticNet",
        n_estimators=500,
        max_samples=0.1,
        bootstrap=True,
        n_jobs=1,
        oob_score=True,
        max_features=1.0,
        bootstrap_features=True,
        test_on="valid_data",
    ):
        if estimator_from == "default":
            model = self.models_dic_base[estimator]
        elif estimator_from == "ensemble":
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        self.bagging_model = BaggingRegressor(
            model,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            oob_score=oob_score,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
        )

        self.bagging_model.fit(self.X_train, self.y_train)
        self.estimator_from_dic["bagging"] = self.bagging_model

        print(f"oob_score - {self.bagging_model.oob_score_}")

        if test_on == "valid_data":
            return self.result_test_df(self.bagging_model).iloc[:, :8]
        if test_on == "test_data":
            return self.result_test_df(self.bagging_model)

    def result_test_df(self, model):
        y_pred_valid = model.predict(self.X_valid)
        mean_absolute_error_valid = round(
            mean_absolute_error(self.y_valid, y_pred_valid), 2
        )
        mean_squared_error_valid = round(
            np.sqrt(
                mean_squared_error(
                    self.y_valid,
                    y_pred_valid,
                )
            ),
            2,
        )
        r2_score_valid = round(r2_score(self.y_valid, y_pred_valid), 2)
        median_absolute_error_valid = round(
            median_absolute_error(
                self.y_valid,
                y_pred_valid,
            ),
            2,
        )
        mape_valid = np.abs((y_pred_valid - self.y_valid) / self.y_valid).mean()
        variance_valid = round(np.var(y_pred_valid), 2)

        y_pred_test = model.predict(self.X_test)
        mean_absolute_error_test = round(
            mean_absolute_error(self.y_test, y_pred_test), 2
        )
        mean_squared_error_test = round(
            np.sqrt(mean_squared_error(self.y_test, y_pred_test)), 2
        )
        r2_score_test = round(r2_score(self.y_test, y_pred_test), 2)
        median_absolute_error_test = round(
            median_absolute_error(
                self.y_test,
                y_pred_test,
            ),
            2,
        )
        mape_test = np.abs((y_pred_test - self.y_test) / self.y_test).mean()
        variance_test = round(np.var(y_pred_test), 2)

        try:
            parameters = model.best_estimator_.get_params()
            building_time = model.cv_results_["mean_fit_time"].sum()

        except AttributeError:
            parameters = model.get_params()
            building_time = "|"

        result_test_df = pd.DataFrame(
            {
                f"{model.__class__.__name__}": [
                    parameters,
                    building_time,
                    mean_absolute_error_valid,
                    mean_squared_error_valid,
                    r2_score_valid,
                    median_absolute_error_valid,
                    mape_valid,
                    variance_valid,
                    mean_absolute_error_test,
                    mean_squared_error_test,
                    r2_score_test,
                    median_absolute_error_test,
                    mape_test,
                    variance_test,
                ]
            },
            index=[
                "parameters",
                "building_time",
                "mean_absolute_error_valid",
                "mean_squared_error_valid",
                "r2_score_valid",
                "median_absolute_error_valid",
                "mape_valid",
                "variance_valid",
                "mean_absolute_error_test",
                "mean_squared_error_test",
                "r2_score_test",
                "median_absolute_error_test",
                "mape_test",
                "variance_test",
            ],
        )

        result_test_df = result_test_df.transpose()

        return result_test_df

    def get_pipe(self, estimator_from, estimator=None):
        if estimator:
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if self.Scaler_mark == "on":
            self.build_pipe = make_pipeline(self.scaler, model)
        if self.PCA_mark == "on" and self.Poly_mark == "off":
            self.build_pipe = make_pipeline(self.scaler, self.pca, model)
        if self.PCA_mark == "off" and self.Poly_mark == "on":
            self.build_pipe = make_pipeline(self.scaler, self.poly, model)
        if self.PCA_mark == "on" and self.Poly_mark == "on":
            self.build_pipe = make_pipeline(self.scaler, self.pca, self.poly, model)

        return self.build_pipe

    def plot_tree(self, params={"max_depth": 3}, dpi=300, save=False):
        tree = DecisionTreeRegressor()
        tree.set_params(**params)
        tree.fit(self.X, self.y)

        self.plot_tree_df = pd.DataFrame(
            index=self.X.columns,
            data=np.round(tree.feature_importances_, 2),
            columns=["Feature Importance"],
        ).sort_values("Feature Importance", ascending=False)

        plt.figure(figsize=(12, 8), dpi=dpi)
        class_names = [str(cls) for cls in self.y.unique()]
        plot_tree(
            tree,
            filled=True,
            feature_names=self.X.columns,
            proportion=True,
            rounded=True,
            precision=2,
            class_names=class_names,
            label="root",
        )

        if save:
            plt.savefig("plot_tree.png")

        return self.plot_tree_df.transpose()

    def probability(
        self, estimator_from="basemodel", estimator=None, test_on="valid_data"
    ):
        if estimator:
            model = self.estimator_from_dic[estimator_from][estimator]
        else:
            model = self.estimator_from_dic[estimator_from]

        if test_on == "valid_data":
            X = self.X_valid
            y = self.y_valid
        if test_on == "test_data":
            X = self.X_test
            y = self.y_test

        else:
            test_res = test_on

        try:
            # the graph should not have a structure
            predictions = model.predict(X)
            test_res = y - predictions
            sns.scatterplot(x=y, y=test_res)
            plt.axhline(y=0, color="r", linestyle="--")
        except:
            pass

        # probability graph
        fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
        _ = sp.stats.probplot(test_res, plot=ax)

        plt.show()
