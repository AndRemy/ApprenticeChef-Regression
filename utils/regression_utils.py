import pandas as pd
import numpy as np
import sklearn.feature_selection as fs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class Utils:
    """
    Utility class for the Apprentice Chef project to help build supervised learning models
    """
    __TEST_SIZE = 0.25
    __SEED = 222

    @classmethod
    def split_execute_model(
            cls,
            x_col: list,
            y_col: str,
            data: pd.DataFrame,
            model: object = LinearRegression(),
            standardize: bool = False,
            test_size: float = __TEST_SIZE,
            seed: float = __SEED
    ) -> dict:
        """
        Function that executes a specific model and returns the train score, test score, and RSME. This function splits
        the data in train and test data sets before performing the execution. Optionally, it performs data
        standardization using sklearn's the StandarScaler.
        
        :param x_col: Name of the columns that conforms the predictors
        :param y_col: Name of the column that will be predicted by the model
        :param data: Data set with the data to train and test the model.
        :param model: Default sklearn.linear_model.LinearRegression(). Instantiated sklearn object to be executed.
        :param standardize: True if you want this method to standardize the data, False if not. Default value is False.
        :param test_size: Default 0.25. The size in percentage of the test data set.
        :param seed: Default 222. The seed used to replicate the randomization in the split of train and test datasets.

        :return: Returns a dictionary with the Train score (train_score), Test score (test_score), and RSME (rsme) in
            the following format:
            {
                "train_score": VALUE,
                "test_score":  VALUE,
                "rsme": VALUE
            }
        """

        x_df = data.loc[:, x_col]
        y_df = data.loc[:, y_col]

        if standardize:
            scale = StandardScaler()
            x_df = scale.fit_transform(x_df)

        x_train, x_test, y_train, y_test = train_test_split(
            x_df,
            y_df,
            test_size=test_size,
            random_state=seed,
        )

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        train_score = np.round(model.score(x_train, y_train), decimals=3)
        test_score = np.round(model.score(x_test, y_test), decimals=3)
        rsme = np.round(mean_squared_error(y_test, y_predict, squared=False), decimals=3)
        # rsme = math.sqrt(np.mean((y_predict - y_test) ** 2))

        return {
            "train_score": train_score,
            "test_score": test_score,
            "rsme": rsme
        }

    @classmethod
    def execute_model(
        cls,
        x_train: list,
        y_train: str,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model: object = LinearRegression(),
    ) -> dict:
        """
        Function that executes a specific model and returns the train score, test score, and RSME. This function
        expects data already split in train and test data sets.

        :param x_train: Train dataset containing the data of the independent variables.
        :param y_train: Train dataset containing the data of the dependent variables.
        :param x_test: Test dataset containing the data of the independent variables.
        :param y_test: Test dataset containing the data of the dependent variables.
        :param model: Default sklearn.linear_model.LinearRegression(). Instantiated sklearn object to be executed.

        :return: Returns a dictionary with the Train score (train_score), Test score (test_score), and RSME (rsme) in
            the following format:
            {
                "train_score": VALUE,
                "test_score":  VALUE,
                "rsme": VALUE
            }
        """

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        train_score = np.round(model.score(x_train, y_train), decimals=3)
        test_score = np.round(model.score(x_test, y_test), decimals=3)
        rsme = np.round(mean_squared_error(y_test, y_predict, squared=False), decimals=3)

        return {
            "train_score": train_score,
            "test_score": test_score,
            "rsme": rsme
        }

    @classmethod
    def stat_feature_selection_regression(
            cls,
            data_df: pd.DataFrame,
            dependent_variable: str,
            features_dict: dict,
            confidence_threshold: float = 0.05,
            correlation_threshold: float = 0.4,
            correlation_method: str = 'pearson'
    ) -> list:
        """
        A function that returns a list of features that have a higher impact on
        a regression model based on two statistical tests:

        - ANOVA Test for categorical and binary features.
            The ANOVA test expects a confidence threshold to determine which features
            are statistical significant. If a 95% threshold is desired, the threshold
            that is expected is 5% (1 - the confidence level). This function will
            select all the features for which the ANOVA test return a p-value below
            the confidence threshold.

        - Correlation for continuous and discrete features.
            The correlation test will calculate a correlation coefficient using the
            method specified on the function parameter. This function expects a
            correlation threshold, which will be used to select the features that
            has a pearson correlation coefficient greater than the specified
            coefficient.

        :param data_df: A DataFrame with the data that is going to be evaluated.
        :param dependent_variable: A string with the name of the column that correspond to the dependent variables.
        :param features_dict: A dictionary that contains the name of the column that correspond to the dependent
            variables.
        :param confidence_threshold: Default 0.05 (confidence of 95%). Threshold used to determine what
            categorical/binary variables will be selected in the ANOVA test. This should be 1 - desired confidence level
        :param correlation_threshold: Default 0.4 (40%). Threshold used to determine what continuous or discrete
            variables will be selected in the correlation analysis. It will select all the features that have a correlation
            (in absolute value) higher than this threshold.
        :param correlation_method: Default 'pearson'. Type of correlation coefficient to be calculated.

        :return: The list of features that were selected under the thresholds.
        """
        # Selecting Categorical and Binary Features
        features = features_dict['categorical'] + features_dict['binary']

        if 'flags' in features_dict.keys():
            features += features_dict['flags']

        results = {
            "feature": [],
            "F_score": [],
            "p_val": []
        }
        for feat in features:
            # The evaluation is made backwards on purpose.
            # The 'y' is the independent categorical variable and the 'X' the dependent continuous variable
            F_score, p_val = fs.f_classif(
                y=data_df[feat],
                X=data_df[[dependent_variable]]
            )

            results["feature"] += [feat]
            results["F_score"] += [F_score[0]]
            results["p_val"] += [p_val[0]]

        results_df = pd.DataFrame(results)
        categorical_feats = list(results_df.loc[results_df['p_val'] < confidence_threshold, 'feature'])

        # Selecting Discrete and Continuous Features
        features = [dependent_variable] + features_dict['discrete'] + features_dict['continuous']

        corr = data_df[features].corr(correlation_method)
        numerical_feats = list(
            corr[(corr[dependent_variable]) < 1 & (corr[dependent_variable] > correlation_threshold)].index)

        return categorical_feats + numerical_feats

