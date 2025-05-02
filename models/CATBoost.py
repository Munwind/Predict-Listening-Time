import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd

pd.options.mode.copy_on_write = True
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

# from lightgbm import LGBMRegressor
import lightgbm as lgb


def process_combinations_fast(df, columns_to_encode, pair_size, max_batch_size=2000):
    # Precompute string versions of all columns once
    str_df = df[columns_to_encode]
    le = LabelEncoder()
    str_df = str_df.astype(str)
    total_new_cols = 0

    for r in pair_size:
        print(f"Processing {r}-combinations")

        # Count total combinations for this r-value
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations to process: {n_combinations}")

        # Process combinations in batches to manage memory
        combos_iter = combinations(columns_to_encode, r)
        batch_cols = []
        batch_names = []

        with tqdm(total=n_combinations) as pbar:
            while True:
                # Collect a batch of combinations
                batch_cols.clear()
                batch_names.clear()

                # Fill the current batch
                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break

                if not batch_cols:  # No more combinations
                    break

                # Process this batch vectorized
                for i, (cols, new_name) in enumerate(zip(batch_cols, batch_names)):
                    # Fast vectorized concatenation
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += '' + str_df[col]

                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)

                total_new_cols += len(batch_cols)
                if len(batch_cols) == max_batch_size:  # Only print on full batches
                    print(f"Progress: {total_new_cols}/{n_combinations} combinations processed")

        print(f"Completed all {r}-combinations. Total columns now: {len(df.columns)}")

    return df


TARGET = 'Listening_Time_minutes'
# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_train.drop(columns=['id'], inplace=True)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_test.drop(columns=['id'], inplace=True)

original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')

original_clean = original.dropna(subset=[TARGET]).drop_duplicates()
df_train = pd.concat([df_train, original_clean], axis=0, ignore_index=True)

df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# df.drop(columns=['id'], inplace=True)
df = df.drop_duplicates()

# outlier removal
df['Episode_Length_minutes'] = np.maximum(0, np.minimum(120, df['Episode_Length_minutes']))
df['Host_Popularity_percentage'] = np.maximum(20, np.minimum(100, df['Host_Popularity_percentage']))
df['Guest_Popularity_percentage'] = np.maximum(0, np.minimum(100, df['Guest_Popularity_percentage']))
df['Host_Popularity_bin'] = pd.cut(df['Host_Popularity_percentage'], bins=[20, 40, 60, 80, 100], labels=[1, 2, 3, 4])
df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

# Encode categorical features
day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['Publication_Day'] = df['Publication_Day'].map(day_mapping)

time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
df['Publication_Time'] = df['Publication_Time'].map(time_mapping)

sentiment_map = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True)
df['Episode_Title'] = df['Episode_Title'].astype('int')
df['Title_Episode_Length'] = df['Episode_Title'] / (df['Episode_Length_minutes'] + 1)
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col]) + 1

# Some Feature engineering
for col in ['Episode_Length_minutes']:
    df[[col + '_sqrt', col + '_squared']] = np.column_stack([
        np.sqrt(df[col]),
        df[col] ** 2
    ])

for col in tqdm(['Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name', 'Episode_Title',
                 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads']):
    df[col + '_EP'] = df.groupby(col)['Episode_Length_minutes'].transform('mean')

df = process_combinations_fast(df, ['Episode_Length_minutes', 'Episode_Title', 'Publication_Time',
                                    'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment',
                                    'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage'],
                               [2, 3, 5, 7], 1000)  # [2, 3, 5, 7]

df = df.astype('float32')

df_train = df.iloc[:-len(df_test)]
df_test = df.iloc[-len(df_test):].reset_index(drop=True)

df_train = df_train[df_train['Listening_Time_minutes'].notnull()]

target = df_train.pop('Listening_Time_minutes')
df_test.pop('Listening_Time_minutes')

df_train.shape, df_test.shape

## Save and load model by pickle
"""
import pickle

# Suppose `model` is your trained LightGBM (or scikit-learn) estimator

# 1) Save to file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 2) Later—load from file
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 3) Use it exactly like the original
y_pred = loaded_model.predict(X_new)

"""

import pickle
import gc

import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
import pickle

# seed and CV
seed = 42
cv = KFold(n_splits=7, random_state=seed, shuffle=True)
pred_test = np.zeros(df_test.shape[0])

# CatBoost params
catboost_params = {
    'iterations': 1_000_00,  # Large number, use early stopping
    'learning_rate': 0.02,  # You can experiment with decaying manually
    'depth': 8,  # CatBoost max_depth is 'depth'
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': seed,
    'verbose': 500,
    'early_stopping_rounds': 30,
    'task_type': 'GPU',  # Use 'CPU' if GPU not available
    'od_type': 'Iter',
}

all_histories = []
features = df_train.columns.tolist()

for fold, (trn_idx, val_idx) in enumerate(cv.split(df_train), 1):
    print(f"Starting Fold {fold}")
    X_trn, y_trn = df_train.iloc[trn_idx].copy(), target.iloc[trn_idx]
    X_val, y_val = df_train.iloc[val_idx].copy(), target.iloc[val_idx]
    X_sub = df_test[X_trn.columns.tolist()].copy()

    # === Target Encoding ===
    encoder = TargetEncoder(n_folds=5, seed=seed, stat="mean")
    print(f"Fold {fold}: Applying Target Encoding...")

    # first 20 new cols
    for col in tqdm(features[:20], desc=f"Fold {fold} TE‐add"):
        # Ensure the column exists in the dataframes before encoding
        if col in X_trn.columns:
            X_trn[f"{col}_te"] = encoder.fit_transform(X_trn[[col]], y_trn)
            X_val[f"{col}_te"] = encoder.transform(X_val[[col]])
            X_sub[f"{col}_te"] = encoder.transform(X_sub[[col]])
        else:
            print(f"Warning: Column '{col}' not found in training data for TE-add.")

    # remaining, in‐place
    for col in tqdm(features[20:], desc=f"Fold {fold} TE‐replace"):
        # Ensure the column exists in the dataframes before encoding
        if col in X_trn.columns:
            X_trn[col] = encoder.fit_transform(X_trn[[col]], y_trn)
            X_val[col] = encoder.transform(X_val[[col]])
            X_sub[col] = encoder.transform(X_sub[[col]])
        else:
            print(f"Warning: Column '{col}' not found in training data for TE-replace.")

    # Create CatBoost pools
    train_pool = Pool(X_trn, label=y_trn)
    valid_pool = Pool(X_val, label=y_val)

    print(f"Fold {fold}: Training CatBoost model...")
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # Predict on test
    test_pred = model.predict(X_sub)
    pred_test += np.clip(test_pred, 0, 120)

    print(f"Fold {fold} finished, best_iteration={model.get_best_iteration()}")
    print("-" * 60)

    del model
    del encoder
    del X_trn, y_trn, X_val, y_val, X_sub
    gc.collect()

# average over folds
pred_test /= cv.n_splits

print("Training complete. Test predictions averaged across folds.")

pred_test

df_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_sub.Listening_Time_minutes = pred_test
df_sub.to_csv('submission_2.csv', index=False)