import pandas as pd
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
    def execute_model(
        cls,
        x_col: list,
        y_col: str,
        data: pd.DataFrame,
        model: object = LinearRegression(),
        standardize: bool = False
    ):
        """
        Executes a model by separating the data in TRAIN and TEST set. It prints the train and test score of the
        created model. Since the model is sent as an object (by reference) to this method, the detailes of the created
        model can be accessed by the program that calls this method.
        :param x_col: Name of the columns that conforms the predictors
        :param y_col: Name of the column that will be predicted by the model
        :param data: Data set with the data to train and test the model.
        :param model: Name of the model to run. By default it will use run the OLS model
        (scikit-learn's LinearRegressison).
        :param standardize: True if you want this method to standardize the data, False if not. Default value is False.
        :return: Doesn't return any information.
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

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        rsme = mean_squared_error(y_test, y_predict, squared=False)  # math.sqrt(np.mean((y_predict - y_test) ** 2))

        print(f"Train Score: {train_score}")
        print(f"Test Score: {test_score}")
        print(f"RSME: {rsme}")