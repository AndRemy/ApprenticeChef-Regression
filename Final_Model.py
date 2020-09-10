# timeit

# Student Name : Marco Andre Remy Silva
# Cohort       : 2 - Haight

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import numpy as np
import pandas as pd

from sklearn.linear_model import Lars
from sklearn.model_selection import train_test_split


################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel("./data/Apprentice_Chef_Dataset.xlsx")


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

# FLAGGING OUTLIERS

# CROSS_SELL_SUCCESS
# NO OUTLIER


# TOTAL_MEALS_ORDERED
out_total_meals_ordered = 350

original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition = original_df.loc[0:, 'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > out_total_meals_ordered]
original_df['out_TOTAL_MEALS_ORDERED'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# UNIQUE_MEALS_PURCH
out_unique_meals_purch = 12

original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition = original_df.loc[0:, 'out_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > out_unique_meals_purch]
original_df['out_UNIQUE_MEALS_PURCH'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# CONTACTS_W_CUSTOMER_SERVICE
out_contacts_w_customer_service = 12.5

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = original_df.loc[0:, 'out_CONTACTS_W_CUSTOMER_SERVICE'][
    original_df['CONTACTS_W_CUSTOMER_SERVICE'] > out_contacts_w_customer_service
]
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# PRODUCT_CATEGORIES_VIEWED
# NO OUTLIER


# AVG_TIME_PER_SITE_VISIT
out_avg_time_per_site_visit = 190

original_df["out_AVG_TIME_PER_SITE_VISIT"] = 0
condition = original_df.loc[0:, 'out_AVG_TIME_PER_SITE_VISIT'][
    original_df['AVG_TIME_PER_SITE_VISIT'] > out_avg_time_per_site_visit
]
original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# MOBILE_NUMBER
# NO OUTLIER


# CANCELLATIONS_BEFORE_NOON
out_cancellation_before_noon = 5

original_df["out_CANCELLATIONS_BEFORE_NOON"] = 0
condition = original_df.loc[0:, 'out_CANCELLATIONS_BEFORE_NOON'][
    original_df['CANCELLATIONS_BEFORE_NOON'] > out_cancellation_before_noon
]
original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# CANCELLATIONS_AFTER_NOON
out_cancellation_after_noon = 0

original_df["out_CANCELLATIONS_AFTER_NOON"] = 0
condition = original_df.loc[0:, 'out_CANCELLATIONS_AFTER_NOON'][
    original_df['CANCELLATIONS_AFTER_NOON'] > out_cancellation_after_noon
]
original_df['out_CANCELLATIONS_AFTER_NOON'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# TASTES_AND_PREFERENCES
# NO OUTLIER


# MOBILE_LOGINS
# NO OUTLIER


# PC_LOGINS
# NO OUTLIER


# WEEKLY_PLAN
out_weekly_plan = 31

original_df["out_WEEKLY_PLAN"] = 0
condition = original_df.loc[0:, 'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > out_weekly_plan]
original_df['out_WEEKLY_PLAN'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# EARLY_DELIVERIES
out_early_deliveries = 7

original_df["out_EARLY_DELIVERIES"] = 0
condition = original_df.loc[0:, 'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > out_early_deliveries]
original_df['out_EARLY_DELIVERIES'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# LATE_DELIVERIES
out_late_deliveries = 8

original_df["out_LATE_DELIVERIES"] = 0
condition = original_df.loc[0:, 'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > out_late_deliveries]
original_df['out_LATE_DELIVERIES'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# PACKAGE_LOCKER
# NO OUTLIER


# REFRIGERATED_LOCKER
# NO OUTLIER


# FOLLOWED_RECOMMENDATIONS_PCT
# NO OUTLIER


# AVG_PREP_VID_TIME
out_avg_prep_vid_time = 260

original_df["out_AVG_PREP_VID_TIME"] = 0
condition = original_df.loc[0:, 'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > out_avg_prep_vid_time]
original_df['out_AVG_PREP_VID_TIME'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# LARGEST_ORDER_SIZE
out_largest_order_size = 8

original_df["out_LARGEST_ORDER_SIZE"] = 0
condition = original_df.loc[0:, 'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > out_largest_order_size]
original_df['out_LARGEST_ORDER_SIZE'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# MASTER_CLASSES_ATTENDED
# NO OUTLIER


# MEDIAN_MEAL_RATING
# NO OUTLIER


# AVG_CLICKS_PER_VISIT
out_avg_clicks_per_visit = 8

original_df["out_AVG_CLICKS_PER_VISIT"] = 0
condition = original_df.loc[0:, 'out_AVG_CLICKS_PER_VISIT'][
    original_df['AVG_CLICKS_PER_VISIT'] < out_avg_clicks_per_visit]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# TOTAL_PHOTOS_VIEWED
out_total_photos_viewed = 420

original_df["out_TOTAL_PHOTOS_VIEWED"] = 0
condition = original_df.loc[0:, 'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > out_total_photos_viewed]
original_df['out_TOTAL_PHOTOS_VIEWED'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# FLAGGING TRENDS

# CROSS_SELL_SUCCESS
# NO TREND

# TOTAL_MEALS_ORDERED
tot_meals_seg_change = 20

original_df['TOTAL_MEALS_ORDERED_change'] = 0
condition = original_df.loc[0:, 'TOTAL_MEALS_ORDERED_change'][original_df['TOTAL_MEALS_ORDERED'] < tot_meals_seg_change]
original_df['TOTAL_MEALS_ORDERED_change'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# UNIQUE_MEALS_PURCH
unique_meals_purch_change = 1

original_df['UNIQUE_MEALS_PURCH_change'] = 0
condition = original_df.loc[0:, 'UNIQUE_MEALS_PURCH_change'][
    original_df['UNIQUE_MEALS_PURCH'] <= unique_meals_purch_change
]
original_df['UNIQUE_MEALS_PURCH_change'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# CONTACTS_W_CUSTOMER_SERVICE
contacts_w_customer_service_change = 10

original_df['CONTACTS_W_CUSTOMER_SERVICE_change'] = 0
condition = original_df.loc[0:, 'CONTACTS_W_CUSTOMER_SERVICE_change'][
    original_df['CONTACTS_W_CUSTOMER_SERVICE'] <= contacts_w_customer_service_change
]
original_df['CONTACTS_W_CUSTOMER_SERVICE_change'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# PRODUCT_CATEGORIES_VIEWED
# NO TREND


# AVG_TIME_PER_SITE_VISIT
# NO TREND

# MOBILE_NUMBER
# NO TREND


# CANCELLATIONS_BEFORE_NOON
# NO TREND


# CANCELLATIONS_AFTER_NOON
# NO TREND


# TASTES_AND_PREFERENCES
# NO TREND


# MOBILE_LOGINS
# NO TREND


# PC_LOGINS
# NO TREND

# WEEKLY_PLAN
# NO TREND


# EARLY_DELIVERIES
# NO TREND


# LATE_DELIVERIES
# NO TREND


# PACKAGE_LOCKER
# NO TREND


# REFRIGERATED_LOCKER
# NO TREND


# FOLLOWED_RECOMMENDATIONS_PCT
# NO TREND


# AVG_PREP_VID_TIME
# NO TREND


# LARGEST_ORDER_SIZE
# NO TREND


# MASTER_CLASSES_ATTENDED
master_classes_attended_change = 2

original_df["MASTER_CLASSES_ATTENDED_change"] = 0
condition = original_df.loc[0:, 'MASTER_CLASSES_ATTENDED_change'][
    original_df['MASTER_CLASSES_ATTENDED'] > master_classes_attended_change
]
original_df['MASTER_CLASSES_ATTENDED_change'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# MEDIAN_MEAL_RATING#MEDIAN_MEAL_RATING
out_median_meal_rating = 2

original_df["out_MEDIAN_MEAL_RATING"] = 0
condition = original_df.loc[0:, 'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > out_median_meal_rating]
original_df['out_MEDIAN_MEAL_RATING'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# AVG_CLICKS_PER_VISIT
# NO TREND


# TOTAL_PHOTOS_VIEWED
total_photos_viewed_change = 0

original_df['TOTAL_PHOTOS_VIEWED_change_1'] = 0
condition = original_df.loc[0:, 'TOTAL_PHOTOS_VIEWED_change_1'][
    original_df['TOTAL_PHOTOS_VIEWED'] == total_photos_viewed_change
]
original_df['TOTAL_PHOTOS_VIEWED_change_1'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)

total_photos_viewed_change = 1000

original_df['TOTAL_PHOTOS_VIEWED_change_2'] = 0
condition = original_df.loc[0:, 'TOTAL_PHOTOS_VIEWED_change_2'][
    original_df['TOTAL_PHOTOS_VIEWED'] > total_photos_viewed_change
]
original_df['TOTAL_PHOTOS_VIEWED_change_2'].replace(
    to_replace=condition,
    value=1,
    inplace=True
)


# DROPPING REVENUE OUTLIERS
original_df = original_df[original_df["REVENUE"] < 6000]

# MAKING A LOGARITHMIC TRANSFORMATION TO IMPROVE PREDICTION
original_df["log_REVENUE"] = np.log(original_df["REVENUE"])

################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

SEED = 222
TEST_SIZE = 0.25

x_variables = [
    'TOTAL_MEALS_ORDERED',                 
    'UNIQUE_MEALS_PURCH',                  
    'CONTACTS_W_CUSTOMER_SERVICE',         
    'AVG_TIME_PER_SITE_VISIT',             
    'PC_LOGINS',                           
    'WEEKLY_PLAN',                         
    'PACKAGE_LOCKER',                      
    'AVG_PREP_VID_TIME',                   
    'LARGEST_ORDER_SIZE',                  
    'MASTER_CLASSES_ATTENDED',             
    'MEDIAN_MEAL_RATING',                  
    'TOTAL_PHOTOS_VIEWED',                 
    
    # Outliers
    'out_AVG_TIME_PER_SITE_VISIT',         
    'out_AVG_PREP_VID_TIME',               
    'out_LARGEST_ORDER_SIZE',              
    'out_AVG_CLICKS_PER_VISIT',            
    'out_TOTAL_PHOTOS_VIEWED',             
    'out_MEDIAN_MEAL_RATING',
    
    # Trends
    'TOTAL_MEALS_ORDERED_change',          
    'UNIQUE_MEALS_PURCH_change',           
    'CONTACTS_W_CUSTOMER_SERVICE_change',  
    'MASTER_CLASSES_ATTENDED_change',      
    'TOTAL_PHOTOS_VIEWED_change_1',        
    'TOTAL_PHOTOS_VIEWED_change_2'         
]

X = original_df.loc[:, x_variables]
y = original_df.loc[:, "log_REVENUE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model

final_model = Lars()
final_model.fit(X_train, y_train)
y_predict = final_model.predict(X_test)


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = final_model.score(X_test, y_test)
