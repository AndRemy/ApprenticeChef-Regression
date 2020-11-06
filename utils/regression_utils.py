import pandas as pd
import numpy as np
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
        standardize: bool = False
    ) -> dict:
        """
        Executes a model by separating the data in TRAIN and TEST set. It prints the train and test score of the
        created model. Since the model is sent as an object (by reference) to this method, the details of the created
        model can be accessed by the program that calls this method.
        
        :param x_col: Name of the columns that conforms the predictors
        :param y_col: Name of the column that will be predicted by the model
        :param data: Data set with the data to train and test the model.
        :param model: Name of the model to run. By default it will use run the OLS model
        (scikit-learn's LinearRegression).
        :param standardize: True if you want this method to standardize the data, False if not. Default value is False.

        :return: Returns a dictionary with the Train score (train_score), Test score (test_score), and RSME (rsme).
        """

        x_df = data.loc[:, x_col]
        y_df = data.loc[:, y_col]

        if standardize:
            scale = StandardScaler()
            x_df = scale.fit_transform(x_df)

        x_train, x_test, y_train, y_test = train_test_split(
            x_df,
            y_df,
            test_size=cls.__TEST_SIZE,
            random_state=cls.__SEED
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

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param model:
        :return: Returns a dictionary with the Train score (train_score), Test score (test_score), and RSME (rsme).
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