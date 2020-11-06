################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LassoCV
from utils.regression_utils import Utils


################################################################################
# Load Data
################################################################################

# Loading data
data_df = pd.read_excel("data/Apprentice_Chef_Dataset.xlsx")

# Removing inconsistencies
data_df = data_df[data_df['LARGEST_ORDER_SIZE'] > 0]

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# Transforming dependent features
data_df['ln_REVENUE'] = np.log(data_df['REVENUE'])


# Transforming independent features
data_df['ln_TOTAL_PHOTOS_VIEWED'] = data_df['TOTAL_PHOTOS_VIEWED']
data_df.loc[data_df['TOTAL_PHOTOS_VIEWED'] != 0, 'ln_TOTAL_PHOTOS_VIEWED'] = data_df['TOTAL_PHOTOS_VIEWED'].apply(np.log)

data_df['ln_UNIQUE_MEALS_PURCH'] = data_df['UNIQUE_MEALS_PURCH']
data_df.loc[data_df['UNIQUE_MEALS_PURCH'] != 0, 'ln_UNIQUE_MEALS_PURCH'] = data_df['UNIQUE_MEALS_PURCH'].apply(np.log)

data_df['ln_AVG_PREP_VID_TIME'] = data_df['AVG_PREP_VID_TIME']
data_df.loc[data_df['AVG_PREP_VID_TIME'] != 0, 'ln_AVG_PREP_VID_TIME'] = data_df['AVG_PREP_VID_TIME'].apply(np.log)

data_df['ln_TOTAL_MEALS_ORDERED'] = data_df['TOTAL_MEALS_ORDERED']
data_df.loc[data_df['TOTAL_MEALS_ORDERED'] != 0, 'ln_TOTAL_MEALS_ORDERED'] = data_df['TOTAL_MEALS_ORDERED'].apply(np.log)


# Flagging trends
data_df['tre_AVG_CLICKS_PER_VISIT'] = 0
data_df.loc[data_df['AVG_CLICKS_PER_VISIT'] > 11, 'tre_AVG_CLICKS_PER_VISIT'] = 1

data_df['tre_LARGEST_ORDER_SIZE'] = 0
data_df.loc[data_df['LARGEST_ORDER_SIZE'] > 8, 'tre_LARGEST_ORDER_SIZE'] = 1

data_df['tre_MEDIAN_MEAL_RATING'] = 0
data_df.loc[data_df['MEDIAN_MEAL_RATING'] > 3, 'tre_MEDIAN_MEAL_RATING'] = 1

data_df['tre_CONTACTS_W_CUSTOMER_SERVICE'] = 0
data_df.loc[data_df['CONTACTS_W_CUSTOMER_SERVICE'] > 10, 'tre_CONTACTS_W_CUSTOMER_SERVICE'] = 1

# New features
data_df['SOME'] = 0
data_df.loc[
    (data_df['WEEKLY_PLAN'] > 0) & (data_df['WEEKLY_PLAN'] < 15),
    'SOME'
] = 1

data_df['SINGLE_MEAL'] = 0
data_df.loc[data_df['UNIQUE_MEALS_PURCH'] == 1, 'SINGLE_MEAL'] = 1

################################################################################
# Train/Test Split
################################################################################

features=[
    'tre_AVG_CLICKS_PER_VISIT',
    'ln_TOTAL_PHOTOS_VIEWED',
    'ln_UNIQUE_MEALS_PURCH',
    'SOME',
    'tre_LARGEST_ORDER_SIZE',
    'LARGEST_ORDER_SIZE',
    'CONTACTS_W_CUSTOMER_SERVICE',
    'MASTER_CLASSES_ATTENDED',
    'MEDIAN_MEAL_RATING',
    'ln_TOTAL_MEALS_ORDERED',
    'tre_MEDIAN_MEAL_RATING',
    'tre_CONTACTS_W_CUSTOMER_SERVICE',
    'SINGLE_MEAL',
    'ln_AVG_PREP_VID_TIME'
]
transformed_dependent = 'ln_REVENUE'

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# Model creation
final_model = LassoCV(eps=0.00092)

model_result = Utils.split_execute_model(
    x_col=features,
    y_col=transformed_dependent,
    data=data_df,
    model=final_model
)


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = model_result['test_score']
print(f"Test Score: {test_score}")

################################################################################
# Serializing Object
################################################################################

with open("model/ApprenticeChef_RegressionModel.pkl", "wb") as file:
    pickle.dump(final_model, file)
