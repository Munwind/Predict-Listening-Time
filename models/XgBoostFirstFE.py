import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
warnings.simplefilter('ignore')
TARGET = 'Listening_Time_minutes'
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads', 'Linear_Feature']

# get the dataset
train_dataset = pd.read_csv('/Predict-Listening-Time/playground-series-s5e4/train.csv')
print("Here")
test_dataset = pd.read_csv('/Predict-Listening-Time/playground-series-s5e4/test.csv')
print("There")
original_dataset = pd.read_csv('/Predict-Listening-Time/playground-series-s5e4/podcast_dataset.csv')
print("WTF")
original_dataset = original_dataset.dropna(subset=[TARGET]).drop_duplicates()
train_dataset = pd.concat([train_dataset, original_dataset], axis=0, ignore_index=True)

def feature_eng(df):
    df = df.copy()
    df['Episode_Num'] = df['Episode_Title'].str[8:]
    df['is_weekend']   = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    return df.drop(columns=['Episode_Title'])

train_dataset = feature_eng(train_dataset)
test_dataset = feature_eng(test_dataset)
train_dataset['is_train'] = 1
test_dataset['is_train'] = 0
train_dataset

combined_dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)
print(len(combined_dataset))
combined_dataset.isna().sum(axis=0)

# combined_dataset['Linear_Feature'] = 0.72 * combined_dataset['Episode_Length_minutes']
combined_dataset['Episode_Length_minutes'] = combined_dataset['Episode_Length_minutes'].fillna(combined_dataset['Episode_Length_minutes'].median())
combined_dataset['Guest_Popularity_percentage'] = combined_dataset['Guest_Popularity_percentage'].fillna(combined_dataset['Guest_Popularity_percentage'].median())
combined_dataset['Number_of_Ads'] = combined_dataset['Number_of_Ads'].fillna(combined_dataset['Number_of_Ads'].median())
combined_dataset['Linear_Feature'] = 0.72 * combined_dataset['Episode_Length_minutes']
combined_dataset
ELM = []
for k in range(3):
    col_name = f'ELm_r{k}'
    combined_dataset[col_name] = combined_dataset['Episode_Length_minutes'].round(k)
    ELM.append(col_name)
combined_dataset

encoded_columns = []

selected_comb = [
     ['Episode_Length_minutes', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Number_of_Ads'],    
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Host_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Podcast_Name'],
    ['Episode_Num', 'Podcast_Name'],  
    ['Guest_Popularity_percentage', 'Podcast_Name'],
    ['ELm_r1', 'Episode_Num'],
    ['ELm_r1', 'Host_Popularity_percentage'], 
    ['ELm_r1', 'Guest_Popularity_percentage'],
    ['ELm_r2', 'Episode_Num'],
    ['ELm_r2', 'Episode_Sentiment'],
    ['ELm_r2', 'Publication_Day'],
    ['Linear_Feature', 'Number_of_Ads'],
    ['Linear_Feature', 'Genre'],
    ['Linear_Feature', 'Episode_Sentiment'],

    
    # 3-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],   
    ['ELm_r1', 'Number_of_Ads', 'Episode_Sentiment'],
    ['ELm_r2', 'Number_of_Ads', 'Podcast_Name'],
    
    # 4-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Genre'],    
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Genre'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Time', 'Podcast_Name'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time', 'Podcast_Name'],
    
]

for comb in selected_comb:
    name = '_'.join(comb)
        
    if len(comb) == 2:
        combined_dataset[name] = combined_dataset[comb[0]].astype(str) + '_' + combined_dataset[comb[1]].astype(str)
    elif len(comb) == 3:
        combined_dataset[name] = (combined_dataset[comb[0]].astype(str) + '_' +
                               combined_dataset[comb[1]].astype(str) + '_' +
                               combined_dataset[comb[2]].astype(str))
    elif len(comb) == 4:
        combined_dataset[name] = (combined_dataset[comb[0]].astype(str) + '_' +
                               combined_dataset[comb[1]].astype(str) + '_' +
                               combined_dataset[comb[2]].astype(str) + '_' +
                               combined_dataset[comb[3]].astype(str))

    encoded_columns.append(name)

combined_dataset[encoded_columns] = combined_dataset[encoded_columns].astype('category')


combined_dataset


train_dataset = combined_dataset[combined_dataset['is_train'] == 1]
test_dataset = combined_dataset[combined_dataset['is_train'] == 0]

train_dataset = train_dataset.drop(columns=['is_train'])
test_dataset = test_dataset.drop(columns=['is_train', 'Listening_Time_minutes'])


def target_encode(df_train, df_val, col, target, stats='mean', prefix='TE'):
    df_val = df_val.copy()
    agg = df_train.groupby(col)[target].agg(stats)    
    if isinstance(stats, (list, tuple)):
        for s in stats:
            colname = f"{prefix}_{col}_{s}"
            df_val[colname] = df_val[col].map(agg[s]).astype(float)
            # df_val[colname].fillna(agg[s].mean(), inplace=True)
    else:
        suffix = stats if isinstance(stats, str) else stats.__name__
        colname = f"{prefix}_{col}_{suffix}"
        df_val[colname] = df_val[col].map(agg).astype(float)
        df_val[colname].fillna(agg.mean(), inplace=True)
    return df_val

y = train_dataset['Listening_Time_minutes']


class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Out‑of‑fold **mean‑rank** encoder with optional smoothing.
    • Encodes each category by the *rank* of its target mean within a fold.
    • Unseen categories get the global mean rank (or −1 if you prefer).
    """
    def __init__(self, cat_cols=None, n_splits=5, smoothing=0):
        self.cat_cols   = cat_cols
        self.n_splits   = n_splits
        self.smoothing  = smoothing       # 0 = no smoothing
        self.maps_      = {}              # per‑fold maps
        self.global_map = {}              # fit on full data for test set

    def _make_fold_map(self, X_col, y):
        means = y.groupby(X_col, dropna=False).mean()
        if self.smoothing > 0:
            counts = y.groupby(X_col, dropna=False).count()
            smooth = (counts * means + self.smoothing * y.mean()) / (counts + self.smoothing)
            means  = smooth
        return {k: r for r, k in enumerate(means.sort_values().index)}

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        kf = KFold(self.n_splits, shuffle=True, random_state=42)
        self.maps_ = {col: [None]*self.n_splits for col in self.cat_cols}

        for fold, (tr_idx, _) in enumerate(kf.split(X)):
            X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
            for col in self.cat_cols:
                self.maps_[col][fold] = self._make_fold_map(X_tr[col], y_tr)

        for col in self.cat_cols:
            self.global_map[col] = self._make_fold_map(X[col], y)

        return self

    def transform(self, X, y=None, fold=None):
        """
        • During CV pass fold index to use fold‑specific maps (leak‑free).
        • At inference time (fold=None) uses global map.
        """
        X = X.copy()
        tgt_maps = {col: (self.global_map[col] if fold is None else self.maps_[col][fold])
                    for col in self.cat_cols}
        for col, mapping in tgt_maps.items():
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X

encode_stats = ['mean']
FOLDS = 10
outer_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
FEATURES = NUMS + CATS + encoded_columns

oof = np.zeros(len(train_dataset))
pred = np.zeros(len(test_dataset))

for fold, (tr_idx, vl_idx) in enumerate(outer_kf.split(train_dataset), 1):
    print(f'\n--- Outer Fold {fold} / {FOLDS} ---')

    # Split train/validation
    X_tr_raw = train_dataset.loc[tr_idx, FEATURES].reset_index(drop=True)
    y_tr     = train_dataset.loc[tr_idx, TARGET].reset_index(drop=True)
    X_vl_raw = train_dataset.loc[vl_idx, FEATURES].reset_index(drop=True)
    y_vl     = train_dataset.loc[vl_idx, TARGET].reset_index(drop=True)
    X_ts_raw = test_dataset[FEATURES].copy()

    # Deep copies for transformation
    X_tr = X_tr_raw.copy()
    X_vl = X_vl_raw.copy()
    X_ts = X_ts_raw.copy()

    # Inner KFold for target encoding
    inner_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for inner_fold, (in_tr_idx, in_vl_idx) in enumerate(inner_kf.split(X_tr_raw)):
        in_tr = pd.concat([X_tr_raw.loc[in_tr_idx].reset_index(drop=True),
                           y_tr.loc[in_tr_idx].reset_index(drop=True)], axis=1)
        in_vl = X_tr_raw.loc[in_vl_idx].reset_index(drop=True)

        for col in encoded_columns:
            for stat in encode_stats:
                te_tmp = target_encode(in_tr, in_vl.copy(), col, TARGET, stats=stat, prefix='TE')
                te_col = f'TE_{col}_{stat}'
                X_tr.loc[in_vl_idx, te_col] = te_tmp[te_col].values

    # Target encode validation and test data
    for col in encoded_columns:
        for stat in encode_stats:
            X_vl = target_encode(pd.concat([X_tr_raw, y_tr], axis=1),
                                 X_vl, col, TARGET, stats=stat, prefix='TE')
            X_ts = target_encode(pd.concat([X_tr_raw, y_tr], axis=1),
                                 X_ts, col, TARGET, stats=stat, prefix='TE')

    # Drop original categorical columns
    X_tr.drop(columns=encoded_columns, inplace=True)
    X_vl.drop(columns=encoded_columns, inplace=True)
    X_ts.drop(columns=encoded_columns, inplace=True)

    # Additional encoding
    enc = OrderedTargetEncoder(cat_cols=CATS, n_splits=FOLDS, smoothing=20)
    enc.fit(X_tr, y_tr)
    X_tr[CATS] = enc.transform(X_tr[CATS])
    X_vl[CATS] = enc.transform(X_vl[CATS])
    X_ts[CATS] = enc.transform(X_ts[CATS])

    # Train model
    model = XGBRegressor(
        tree_method='hist',
        n_estimators=50_000,
        max_depth=15,
        learning_rate=0.02,
        colsample_bytree=0.5,
        subsample=0.9,
        random_state=42,
        min_child_weight=10,
        enable_categorical=True,
        eval_metric='rmse',
        early_stopping_rounds=150
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=500
    )

    # Save out-of-fold prediction and test prediction
    oof[vl_idx] = model.predict(X_vl)
    pred += model.predict(X_ts)

# Average test predictions
pred /= FOLDS


xgb_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

pred

test_ids = test_dataset['id'].unique()

test_ids

submit_df = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': pred
})

submit_df.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')


