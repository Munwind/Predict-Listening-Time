import os
import gc
import shap
import math
import ctypes
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from colorama import Fore, Style
from itertools import combinations
from numpy.typing import ArrayLike
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from sklearn.preprocessing import QuantileTransformer

import tensorflow as tf, deeptables as dt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers.legacy import Adam
from deeptables.utils.shap import DeepTablesExplainer
from deeptables.models import DeepTable, ModelConfig, deepnets

warnings.filterwarnings('ignore')
print('TensorFlow version:',tf.__version__+',',
      'GPU =',tf.test.is_gpu_available())
print('DeepTables version:',dt.__version__)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(seed=42)

def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
clean_memory()

def print_memory_usage(X, X_test, wording='default'):
    print(f"Memory usage {wording}      X: {X.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    print(f"Memory usage {wording} X_test: {X_test.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n")

train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
train_orig = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train original shape:", train_orig.shape, '\n')

train = pd.concat([train, train_orig], ignore_index=True).drop_duplicates()
train.dropna(subset=['Listening_Time_minutes'], inplace=True)
train.reset_index(drop=True, inplace=True)
print("Train combied shape:", train.shape)

ELM = []
for k in range(3):
    col_name = f'ELm_r{k}'
    train[col_name] = train['Episode_Length_minutes'].round(k)
    test[col_name] = test['Episode_Length_minutes'].round(k)
    ELM.append(col_name)


def target_encoding(train, target, test=None, feat_to_encode=None, min_samples_leaf=1, smoothing=0.1):
    train.sort_index(inplace=True)
    if feat_to_encode is None:
        feat_to_encode = train.columns.tolist()
    encoder_params = dict(cols=feat_to_encode, min_samples_leaf=min_samples_leaf, smoothing=smoothing)

    oof_parts = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train, target):
        encoder = TargetEncoder(**encoder_params)
        encoder.fit(train.iloc[tr_idx], target.iloc[tr_idx])
        encoded = encoder.transform(train.iloc[val_idx])
        encoded[feat_to_encode] = encoded[feat_to_encode].astype('float32')
        encoded.index = train.index[val_idx]
        oof_parts.append(encoded)

    final_encoder = encoder = TargetEncoder(**encoder_params)
    final_encoder.fit(train, target)
    if test is not None:
        test = final_encoder.transform(test)
        test[feat_to_encode] = test[feat_to_encode].astype('float32')

    train = pd.concat(oof_parts).sort_index()
    return train, test

X = train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = train.Listening_Time_minutes
X_test = test.drop(['id'], axis=1)
del train, test
print("X      shape:", X.shape)
print("X_test shape:", X_test.shape, '\n')

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols), '\n')

# Only 1 missing value to fill
m_ = X['Number_of_Ads'].mode()[0]
X['Number_of_Ads'] = X['Number_of_Ads'].fillna(m_)
# Fill missing values and create an indicator column
for c in num_cols:
    if X[c].isna().any():
        m = X[c].mean()
        X[f'NA_{c}'] = X[c].isna().astype('int8')
        X[c] = X[c].fillna(m)
        X_test[f'NA_{c}'] = X_test[c].isna().astype('int8')
        X_test[c] = X_test[c].fillna(m)
        num_cols.append(f'NA_{c}')

pair_size = [2, 3, 4]
encode_cols = ['Episode_Length_minutes',
               'Number_of_Ads',
               #               'Episode_Title',
               'Episode_Sentiment',
               'Publication_Day',
               'Publication_Time',
               'Podcast_Name',
               #               'Genre',
               'Guest_Popularity_percentage',
               'Host_Popularity_percentage']


def eng_combos(df):
    df_str_np = df[encode_cols].astype(str).values.astype('U')
    encoded_columns = []
    selected_comb = [
        ['Episode_Num', 'Host_Popularity_percentage'],
        ['Episode_Num', 'Guest_Popularity_percentage'],
        ['Episode_Num', 'Number_of_Ads'],
        ['ELm_r1', 'Episode_Num'],
        ['ELm_r1', 'Host_Popularity_percentage'],
        ['ELm_r1', 'Guest_Popularity_percentage'],
        ['ELm_r2', 'Episode_Num'],
        ['ELm_r2', 'Episode_Sentiment'],
        ['ELm_r2', 'Publication_Day'],
        ['ELm_r1', 'Number_of_Ads', 'Episode_Sentiment'],
        ['ELm_r2', 'Number_of_Ads', 'Podcast_Name'],
        ['Episode_Num', 'Podcast_Name'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Episode_Num', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day'],
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
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Time', 'Podcast_Name'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Day'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Genre'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time', 'Genre'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ]

    for comb in selected_comb:
        name = '.' + '_'.join(comb)

        if len(comb) == 2:
            df[name] = df[comb[0]].astype(str) + '_' + df[comb[1]].astype(str)

        elif len(comb) == 3:
            df[name] = (df[comb[0]].astype(str) + '_' +
                        df[comb[1]].astype(str) + '_' +
                        df[comb[2]].astype(str))

        elif len(comb) == 4:
            df[name] = (df[comb[0]].astype(str) + '_' +
                        df[comb[1]].astype(str) + '_' +
                        df[comb[2]].astype(str) + '_' +
                        df[comb[3]].astype(str))

        encoded_columns.append(name)

    df[encoded_columns] = df[encoded_columns].astype('category')
    for r in pair_size:
        for cols in combinations(range(len(encode_cols)), r):
            col_names = [encode_cols[i] for i in cols]
            new_col_name = '._' + '_'.join(col_names)
            concat = df_str_np[:, cols[0]]
            for i in range(1, r):
                concat = np.char.add(np.char.add(concat, '_'), df_str_np[:, cols[i]])
            df[new_col_name] = pd.Categorical(concat)

    return df


def feat_eng(df, num_chunks=4, n_jobs=4):
    df['_Has_Ads'] = (df['Number_of_Ads'] > 0).astype('int8')
    df['_Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype('int8')
    df['_sqrt_Episode_Length_minutes'] = np.sqrt(df['Episode_Length_minutes']).astype('float32')
    df['_squared_Episode_Length_minutes'] = (df['Episode_Length_minutes'] ** 2).astype('float32')
    df['_sin_Episode_Length_minutes'] = np.sin(2 * np.pi * df['Episode_Length_minutes'] / 60).astype('float32')
    df['_cos_Episode_Length_minutes'] = np.cos(2 * np.pi * df['Episode_Length_minutes'] / 60).astype('float32')
    df['_sin_Host_Popularity_percentage'] = np.sin(2 * np.pi * df['Host_Popularity_percentage'] / 20).astype('float32')
    df['_cos_Host_Popularity_percentage'] = np.cos(2 * np.pi * df['Host_Popularity_percentage'] / 20).astype('float32')
    df['Episode_Num'] = df['Episode_Title'].str[8:]
    df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time_enc'] = df['Publication_Time'].replace(time_dict)
    df['_sin_Publication_Time'] = np.sin(2 * np.pi * df['Publication_Time_enc'] / 2).astype('float32')
    df['_cos_Publication_Time'] = np.cos(2 * np.pi * df['Publication_Time_enc'] / 2).astype('float32')
    df = df.drop(['Publication_Time_enc'], axis=1)

    chunks = np.array_split(df, num_chunks)
    results = Parallel(n_jobs=n_jobs)(delayed(eng_combos)(chunk) for chunk in chunks)
    df = pd.concat(results, ignore_index=True)

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    new_enc_cols = [col for col in df.columns if col.startswith('.')]
    return df, new_cat_cols, new_num_cols, new_enc_cols


X, new_cat_cols, new_num_cols, new_enc_cols = feat_eng(X)
X_test, new_cat_cols, new_num_cols, new_enc_cols = feat_eng(X_test)
num_cols += new_num_cols;
cat_cols += new_cat_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols) + 2)  # +2 NA indicator columns
print("len(new_enc_cols):", len(new_enc_cols), '\n')
print_memory_usage(X, X_test, wording='after feat eng')
clean_memory()

# Reduce memory usage
X_all = pd.concat([X, X_test])
for col in X_all.columns:
    if col.startswith('._'):
        X_all[col] = X_all[col].astype('category').cat.codes.astype('int32')
X = X_all.iloc[:len(X)];
X_test = X_all.iloc[len(X):]
print_memory_usage(X, X_test, wording='after reduce')
del X_all;
clean_memory()

X, X_test = target_encoding(X, y, X_test, feat_to_encode=new_enc_cols)
print_memory_usage(X, X_test, wording='after encode')
clean_memory()

scaler = QuantileTransformer(subsample=10 ** 9)
X[num_cols] = scaler.fit_transform(X[num_cols]).astype(np.float32)
X_test[num_cols] = scaler.transform(X_test[num_cols]).astype(np.float32)
print_memory_usage(X, X_test, wording='after scale')
clean_memory()

num_cols += new_enc_cols
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols), '\n')

LR_START = 1e-7
LR_MAX = 1e-3
LR_MIN = 1e-7
LR_RAMPUP_EPOCHS = 2
LR_SUSTAIN_EPOCHS = 3
EPOCHS = 9

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel('Epoch'); plt.ylabel('LR')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))
LR_Scheduler = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

class CFG:
    TRAIN = True
    FIT_VERBOSE = 2
    folds = 5
    epochs = 9
    batch_size = 128
    LR_Scheduler = [LR_Scheduler]
    optimizer = Adam(learning_rate=1e-3)

    conf = ModelConfig(auto_imputation=False,
                       auto_discrete=True,
                       fixed_embedding_dim=True,
                       embeddings_output_dim=4,
                       embedding_dropout=0.3,
                       nets=['dnn_nets'],
                       dnn_params={
                           'hidden_units': ((1024, 0.3, True),
                                             (512, 0.3, True),
                                             (256, 0.3, True)),
                           'dnn_activation': 'relu',
                       },
                       autoint_params={
                            'num_attention': 3,
                            'num_heads': 1,
                            'dropout_rate': 0.0,
                            'use_residual': True,
                       },
                       stacking_op='concat',
                       output_use_bias=False,
                       optimizer=optimizer,
                       task='regression',
                       loss='auto',
                       metrics=['RootMeanSquaredError'],
                       earlystopping_patience=1,
                       )

def train_model(X, y, nn=['dnn_nets'], model_n='model_1'):
    print("Data shape:", X.shape, '\n')
    kf = KFold(n_splits=CFG.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []
    for fi, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print("#"*25)
        print(f"### Fold {fi+1}/{CFG.folds} ...")
        print("#"*25)

        K.clear_session()
        CFG.conf = CFG.conf._replace(nets=nn)
        model = DeepTable(config=CFG.conf)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  validation_data=(X.iloc[valid_idx], y.iloc[valid_idx]),
                  callbacks=CFG.LR_Scheduler,
                  batch_size=CFG.batch_size, epochs=CFG.epochs, verbose=CFG.FIT_VERBOSE)
        models.append(model)

        # Save model
        os.makedirs(f"/kaggle/working/models/fold{fi}_{model_n}", exist_ok=True)
        os.makedirs(f"/tmp/workdir/kaggle/working/models/fold{fi}_{model_n}", exist_ok=True)
        model.save(f"/kaggle/working/models/fold{fi}_{model_n}")
        os.system(f"cp -r /kaggle/working/models/fold{fi}_{model_n}/* /tmp/workdir/kaggle/working/models/fold{fi}_{model_n}/")
        # Avoid some errors
        with K.name_scope(CFG.optimizer.__class__.__name__):
            for j, var in enumerate(CFG.optimizer.weights):
                name = 'variable{}'.format(j)
                CFG.optimizer.weights[j] = tf.Variable(var, name=name)
        CFG.conf = CFG.conf._replace(optimizer=CFG.optimizer)

        oof_pred = model.predict(X.iloc[valid_idx], verbose=1, batch_size=512).flatten()
        m = np.round(np.sqrt(np.mean((oof_pred - y.iloc[valid_idx])**2)),4)
        print(f"{Fore.GREEN}{Style.BRIGHT}\nFold {fi+1} | score: {m:.4f}{Style.RESET_ALL}\n")
        oof[valid_idx] = oof_pred
    m_all = np.round(np.sqrt(np.mean((oof - y)**2)),4)
    print(f"{Fore.BLUE}{Style.BRIGHT}Overall CV score: {m_all:.4f}{Style.RESET_ALL}\n")
    display(plot_model(model.get_model().model))
    return models, oof

if CFG.TRAIN==True:
    models_1, oof_1 = train_model(X, y, nn=['dnn_nets'], model_n='model_1')
#    models_2, oof_2 = train_model(X, y, nn=['dcn_nets'], model_n='model_2')
#    models_2, oof_2 = train_model(X, y, nn=['pnn_nets'], model_n='model_2')
#    models_2, oof_2 = train_model(X, y, nn=['dnn_nets','autoint_nets'], model_n='model_2')

    if 'models_2' in globals():
        models = models_1 + models_2
        y_pred_ensemble = y.copy()
        y_pred_ensemble['pred'] = np.mean((oof_1, oof_2), axis=0)
        m_ensemble = np.round(np.sqrt(np.mean((y_pred_ensemble['pred'] - y)**2)),4)
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Ensemble CV score: {m_ensemble:.4f}{Style.RESET_ALL}\n")
    else: models = models_1
    clean_memory()

# Load models
def load_model(paths):
    models = []
    for fold in sorted(os.listdir(paths)):
        path = os.path.join(paths, fold)
        for file in os.listdir(path):
            if file.endswith('.h5'):
                models.append(DeepTable.load(path, file))
    return models

models = load_model("/kaggle/working/models")
print("\nmodels:", models)

# Get feature importance
dt_explainer = DeepTablesExplainer(models[0], X_test, num_samples=100)
shap_values = dt_explainer.get_shap_values(X_test[:10], nsamples='auto')

# Plot feature importance
shap.summary_plot(shap_values, X_test, max_display=25, plot_size=(15,8))

# Inference
class AvgModel:
    def __init__(self, models: list[BaseEstimator]):
        self.models = models
    def predict(self, X: ArrayLike):
        preds = []
        for model in self.models:
            pred = model.predict(X, verbose=1, batch_size=512).flatten()
            preds.append(pred)
        return np.mean(preds, axis=0)

avg_model = AvgModel(models)
test_pred = avg_model.predict(X_test)

sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()