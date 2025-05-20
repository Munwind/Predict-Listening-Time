# Kaggle_predict_listening
# Predict Podcast Listening Time

## Kaggle Playground Series – Season 5, Episode 4 (PS‑S5E4)

> *Tabular regression · Root‑Mean‑Squared Error (RMSE)*

![image](https://github.com/user-attachments/assets/44935934-b7e4-41ea-bb4c-78c3f8fc47a2)

---

### 1 · Competition snapshot

The goal is to predict how long a listener will spend on a podcast episode (in **minutes**) from a mix of categorical and numerical metadata such as title, genre, host popularity, publication time, etc. Kaggle evaluates submissions with **RMSE** – lower is better.([kaggle.com](https://www.kaggle.com/competitions/playground-series-s5e4?utm_source=chatgpt.com))

### 2 · Repository layout

| Path                   | Role                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `EDA.ipynb`            | Exploratory data analysis + sanity checks                                           |
| `XgBoostFirstFE.py`    | XGBoost baseline **with heavy feature engineering** (7‑fold OOF)|
| `LightGBM.py`          | GPU LightGBM GBDT model + combo features + target encoding |
| `CATBoost.py`          | GPU CatBoost regressor with the same engineered feature set |
| `DeepNeuralNetwork.py` | DeepTables (TF‑based) wide‑and‑deep network for tabular data |
| `ensemble.ipynb`       | Simple arithmetic mean / rank‑averaging of model predictions |
| `requirements.txt`     | Exact python package set for local runs                           

### 3 · Data pipeline & feature engineering

1. **Dataset augmentation** – the official `train.csv` is concatenated with `podcast_dataset.csv` to enlarge the learning signal.
2. **Basic cleaning** – extreme outliers are clipped to sensible ranges (e.g. `Episode_Length_minutes ∈ [0, 120]`).
3. **Hand‑crafted helpers**

   * Weekend flag & ad‑presence flag
   * Rounded episode length buckets *(ELm\_r0/1/2)*
   * Linear derivative feature `Linear_Feature = 0.72 × Episode_Length_minutes`
4. **High‑order interactions** – >100 pair / triple / quadruple categorical combos are generated, then stored as pandas `category` to keep memory low.
5. **Target encoding** – all newly created combo columns plus raw categoricals are encoded in a **leak‑free, fold‑aware** fashion (mean, rank or RAPIDS’ `TargetEncoder` depending on the script).

### 4 · Models

| Model            | Highlights                               | CV folds | Local CV RMSE\* |
| ---------------- | ---------------------------------------- | -------- | --------------- |
| XGBoost (`hist`) | 50 k trees, depth 15, early stop 150     | 7       | ≈ **11.698**      |
| LightGBM (GPU)   | Learning‑rate decay callback, 512 leaves | 7        | ≈ **11.725**      |
| CatBoost (GPU)   | 100 k iters, depth 8, iter‑based OD      | 7        | ≈ **11.8**      |
| Deep Neural Net  | Wide‑&‑Deep stack (`dnn_nets`), cyc. LR  | 5        | ≈ **11.78**      |


The final submission is the **simple mean** (ensemble) of the four model predictions, which scored **≈ 11.793 RMSE on the public LB** (top 7.09%) and **≈ 11.697 on the private LB** (top 6.12%).

### 5 · Reproduce my results

> All paths in the scripts follow the Kaggle notebook directory layout. If you are running locally, export `DATA_DIR` or tweak the constants at the top of each file.

```bash
# clone
$ git clone https://github.com/<your‑user>/ps‑s5e4‑podcast‑listening‑time.git
$ cd ps‑s5e4‑podcast‑listening‑time

# install deps (CUDA 11+, Python 3.10 recommended)
$ pip install -r requirements.txt

# 1. XGBoost
$ python XgBoostFirstFE.py   # writes submission.csv
# 2. LightGBM  (needs GPU)
$ python LightGBM.py          # writes submission_4.csv
# 3. CatBoost  (needs GPU)
$ python CATBoost.py          # writes submission_2.csv
# 4. Deep NN   (CPU or GPU; slower)
$ python DeepNeuralNetwork.py # writes submission.csv

# 5. Blend inside the notebook
$ jupyter notebook ensemble.ipynb  # or run‑all in Kaggle UI
```
