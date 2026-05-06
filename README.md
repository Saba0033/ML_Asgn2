# IEEE-CIS Fraud Detection

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) - ამოცანა მდგომარეობს ონლაინ ტრანზაქციებში თაღლითობის ამოცნობაში. მონაცემები მოწოდებულია Vesta Corporation-ის მიერ და შედგება ორი წყაროსგან:

- `train_transaction.csv` / `test_transaction.csv` — ტრანზაქციის ჩანაწერი (თანხა, ბარათის ტიპი, საფოსტო კოდი, ელ.ფოსტის დომენი, 339 ანონიმიზებული V-ფიჩერი).
- `train_identity.csv` / `test_identity.csv` — დამატებითი იდენტიფიკაციის ინფორმაცია (მოწყობილობა, ბრაუზერი, IP). მიერთებულია ტრანზაქციების მხოლოდ ~25%-ზე.

გაერთიანების შემდეგ ვიღებთ ~590,000 სტრიქონს და ~434 სვეტს. სამიზნე ცვლადი — `isFraud` (ბინარული, ~3.5% დადებითი კლასი). შეფასების მეტრიკა — ROC-AUC.

GitHub: https://github.com/Saba0033/ML_Asgn2
DagsHub MLflow: https://dagshub.com/Saba0033/ML_Asgn2.mlflow

## ჩემი მიდგომა

თითოეული მოდელის არქიტექტურისთვის შევქმენი ცალკე notebook და ცალკე MLflow ექსპერიმენტი. თითოეულ notebook-ში ვამოწმებ რამდენიმე imputation სტრატეგიას, სამ feature selection მიდგომას და hyperparameter-ების გრიდს, რომელიც განზრახ მოიცავს underfit, well-fit და overfit კონფიგურაციებს. საუკეთესო კონფიგურაციას ვამოწმებ 3-fold cross-validation-ით და შემდეგ ვინახავ Pipeline-ად MLflow-ში.

ყოველი notebook-ის ბოლო ნაბიჯი არის `register_if_better(...)` — ის ადარებს ამ run-ის CV ROC-AUC-ს registry-ში არსებულ `IEEEFraudBestModel`-ის უახლეს ვერსიას და მხოლოდ მაშინ ანაცვლებს, თუ შედეგი უკეთესია. ამის შედეგად ყველა notebook დამოუკიდებელია — შეიძლება ნებისმიერი თანმიმდევრობით გავუშვათ — მაგრამ registry ყოველთვის ინახავს მხოლოდ საუკეთესო ვერსიას. `model_inference.ipynb` პირდაპირ ჩატვირთავს `models:/IEEEFraudBestModel/latest`-ს და რან-ავს raw test-ზე.

ჯამში გავტესტე 7 არქიტექტურა: Logistic Regression (L1), Logistic Regression (L2), Decision Tree, Random Forest, AdaBoost, Gradient Boosting (sklearn HistGB), XGBoost.

## რეპოზიტორიის სტრუქტურა

```
ML_Asgn2/
├── README.md
├── requirements.txt
├── .gitignore
├── results_cache.json                              — per-notebook cached metrics
│
├── 00_eda.ipynb                                    — EDA + plots/-ში გრაფიკები
│
├── model_experiment_LogisticRegression.ipynb       — LogReg L2
├── model_experiment_LogisticRegression_L1.ipynb    — LogReg L1
├── model_experiment_DecisionTree.ipynb
├── model_experiment_RandomForest.ipynb
├── model_experiment_AdaBoost.ipynb
├── model_experiment_GradientBoosting.ipynb
├── model_experiment_XGBoost.ipynb
│
├── model_inference.ipynb                           — registry-დან ჩატვირთვა + submission
│
├── src/
│   ├── data_utils.py                               — load_train, load_test, split_columns
│   ├── preprocessing.py                            — preprocessor-ები + CorrelationPruner
│   └── mlflow_utils.py                             — DagsHub init + named runs + metrics
│
├── data/                                           — Kaggle CSVs (gitignored)
├── plots/                                          — EDA-ს გრაფიკები
└── submissions/                                    — submission.csv
```

ყველა model_experiment notebook ერთი და იგივე სტრუქტურით:

1. Setup
2. Data Loading
3. Cleaning — MLflow runs სხვადასხვა imputation სტრატეგიისთვის
4. Feature Engineering — MLflow run engineered ფიჩერების lift-ისთვის
5. Feature Selection — MLflow runs სამი მიდგომისთვის
6. Training & Hyperparameter Tuning — MLflow run ყოველ HP combo-ზე
7. Cross-Validation of the Best Configuration
8. Final Pipeline + Optional Registration

## EDA

ყველა გრაფიკი იწერება `00_eda.ipynb`-ის გაშვებისას `plots/` დირექტორიაში.

### კლასების ბალანსი

დადებითი კლასი (`isFraud=1`) მხოლოდ ~3.5%. accuracy აქ არ ვარგა — ოპტიმიზაცია ROC-AUC-ზე, მონიტორინგი PR-AUC-ზე.

![Class balance](plots/01_class_balance.png)

### ცარიელი მნიშვნელობები

დატასეტში სვეტების უმეტესობას NaN აქვს, ბევრს — 50%-ზე მეტი. NaN-ის სტრუქტურა თავისთავად სიგნალია, ამიტომ tree მოდელებისთვის ვირჩევ sentinel imputation-ს (`-999`).

![Missing distribution](plots/02_missing_distribution.png)

### TransactionAmt-ის განაწილება

Raw თანხა heavy-tailed; `log1p` ანაწილებს გაცილებით სიმეტრიულად. ამიტომ `TransactionAmt_log` ფიჩერი ემატება ყოველ pipeline-ში.

![Transaction amount](plots/03_transaction_amt.png) ![Amount by class](plots/04_amt_by_class.png)

### კატეგორიული ფიჩერები vs fraud rate

`P_emaildomain`-ისა და `card4`-ის ზოგიერთ კატეგორიაში fraud rate აღწევს საშუალოს ~5x-ს. ეს ამართლებს email TLD-ის (suffix) ფიჩერად გამოყოფას.

![Email domain fraud rate](plots/05_P_emaildomain_fraud_rate.png)

### დროის სტრუქტურა

![Hour of day](plots/06_hour_of_day.png)

### V-ფიჩერების კორელაცია

339 ანონიმიზებული V-ფიჩერი მაღალ-კორელირებულია. ეს ამართლებს correlation-pruning selector-ს feature selection ეტაპზე.

![V correlation](plots/07_v_correlation.png)

## Feature Engineering

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

| მოდელის ტიპი | მიდგომა | რატომ |
|---|---|---|
| ლინეარული (LogReg L1/L2) | OneHotEncoder, `max_categories=20`, `min_frequency=0.001` | ლინეარულ მოდელს თითო კატეგორიის სვეტი ცალ-ცალკე სჭირდება. cap აუცილებელია — `DeviceInfo`/`id_31`-ს 5000+ უნიკალური მნიშვნელობა აქვს. |
| Tree-based | OrdinalEncoder | ხეები ბუნებრივად იმუშავებენ რიცხვით კოდებზე; OHE 430 ფიჩერისგან 5000+ შექმნიდა. |

ორივე pipeline-ი იწყება `_to_string_block` ეტაპით, რომელიც mixed numeric+NaN კატეგორიულ სვეტებს (`card1`, `addr1`) აყოფს უნიფიცირებულ string-ებად — სხვაგვარად encoder ვერ მუშაობს.

### NaN მნიშვნელობების დამუშავება

| სტრატეგია | სად ვტესტავ | run name | მოლოდინი |
|---|---|---|---|
| `median` (numeric) + `'missing'` (categorical) | linear notebook | `LogReg_*_Cleaning_maxcats*` | სტანდარტული |
| `0.0` fill | tree notebook | `*_Cleaning_fill0` | ნეიტრალური; სუსტდება NaN-სიგნალი |
| `-999.0` fill (sentinel) | tree notebook | `*_Cleaning_fill-999` | სავარაუდო გამარჯვებული tree-ისთვის |

### დამატებითი ფიჩერები

ყველა ერთიდაიგივე — `src/preprocessing.engineer_features`-ში:

| ფიჩერი | ფორმულა | მოტივაცია |
|---|---|---|
| `TransactionAmt_log` | `np.log1p(TransactionAmt)` | EDA: heavy-tailed განაწილება |
| `P_emaildomain_suffix` | TLD payer email-ის | EDA: TLD ატარებს fraud სიგნალის უმეტესობას |
| `R_emaildomain_suffix` | TLD recipient email-ის | სიმეტრიულად |

### Cleaning მიდგომები

- `src.data_utils.reduce_mem_usage` — numeric dtype-ების downcast-ი ამცირებს გაერთიანებული train ცხრილის ზომას ~1.4 GB-დან ~600 MB-მდე.
- `load_train(sample_frac=0.3)` — სტრატიფიცირებული sub-sampling ლოკალური ექსპერიმენტებისთვის.
- `load_test` ნორმალიზებს Kaggle-ის `id-XX` (test_identity) → `id_XX` (train_identity) სვეტების სახელებს.

## Feature Selection

თითოეულ notebook-ში სამი selector — თითოეული ცალკე MLflow run-ად, `n_features_kept` მეტრიკით.

| Selector | სად ვიყენებ | ტიპი |
|---|---|---|
| VarianceThreshold | ყველგან — drop near-constant columns | filter |
| CorrelationPruner (custom) | tree notebooks — V-ბლოკი ძლიერ კორელირებულია | filter |
| SelectFromModel(RandomForest) | tree notebooks | embedded |
| SelectFromModel(LogReg L1) | linear notebooks | embedded (კანონიკური GLM-style) |

შერჩევის წესი:

```
selection_score = val_roc_auc - 0.5 * max(0, overfit_gap - 0.02)
```

ანუ ვირჩევთ მაღალი val_AUC-ის მქონეს, მაგრამ ვაჯარიმებთ თუ overfit_gap > 2%. argmax(val_AUC) მარტო რომ გამოვეყენებინა, ხშირად ჩავარდებოდა აშკარად overfit-ულ მოდელზე.

## Training

### ტესტირებული მოდელები

7 არქიტექტურა, თითოეულს ცალკე notebook და ცალკე MLflow ექსპერიმენტი:

| # | არქიტექტურა | MLflow ექსპერიმენტი |
|---|---|---|
| 1 | Logistic Regression (L2) | `LogisticRegression_L2_Training` |
| 2 | Logistic Regression (L1) | `LogisticRegression_L1_Training` |
| 3 | Decision Tree | `DecisionTree_Training` |
| 4 | Random Forest | `RandomForest_Training` |
| 5 | AdaBoost | `AdaBoost_Training` |
| 6 | Gradient Boosting (HistGB) | `GradientBoosting_Training` |
| 7 | XGBoost | `XGBoost_Training` |

### Leaderboard (CV ROC-AUC, 3-fold StratifiedKFold, SAMPLE_FRAC=0.3)

| # | არქიტექტურა | CV val ROC-AUC | CV std | overfit gap | რეჟიმი |
|---|---|---|---|---|---|
| 1 | **XGBoost** | **0.9354** | 0.0040 | 0.0513 | well-fit (champion) |
| 2 | Gradient Boosting (HistGB) | 0.9237 | 0.0046 | 0.0445 | well-fit |
| 3 | Random Forest | 0.8959 | 0.0044 | 0.0572 | სუსტი overfit |
| 4 | Decision Tree | 0.7663 | 0.0032 | -0.0023 | well-fit baseline |

შენიშვნა: LogReg L1, LogReg L2 და AdaBoost notebook-ები იმავე სტრუქტურით აშენებულია და MLflow-ზე ცალკე ექსპერიმენტებშია, თითოეულის HP grid-ი იგივე under/well/overfit ლოგიკას მიყვება. შედეგები DagsHub MLflow UI-ში ხილულია (იხ. ბმული ქვემოთ).

### თითოეული არქიტექტურის საუკეთესო კონფიგურაცია

| არქიტექტურა | საუკეთესო HP combo | best fill | best selector | n_features_kept |
|---|---|---|---|---|
| XGBoost | `n=800, lr=0.03, d=10, ss=0.7, cs=0.6` | 0.0 | correlation | 290 |
| Gradient Boosting | `lr=0.05, max_iter=500, max_depth=8` | 0.0 | correlation | 290 |
| Random Forest | `n=200, max_depth=None, min_samples_leaf=20` | 0.0 | correlation | 290 |
| Decision Tree | `max_depth=6` | 0.0 | correlation | 290 |

ოთხივე tree-based notebook აირჩია `numeric_fill=0` და `CorrelationPruner` (V-block-ის redundancy-ის გამო).

### Hyperparameter ოპტიმიზაცია — under/well/over-fit ცხრილები

დავალების ინსტრუქცია ცხადად ამბობს რომ overfit/underfit რეჟიმების ჩვენება უფრო მნიშვნელოვანია ვიდრე უმაღლესი score. ყველა HP გრიდი მოიცავს ორივე ექსტრემუმს და `selection_score = val_AUC - 0.5·max(0, gap-0.02)` წესით ვირჩევთ ჯანმრთელ კომპრომისს.

**XGBoost** (`XGBoost_Training`):

| n_estimators | learning_rate | max_depth | train AUC | val AUC | gap | რეჟიმი |
|---|---|---|---|---|---|---|
| 100 | 0.30 | 3 | 0.9162 | 0.9072 | +0.0090 | underfit |
| 200 | 0.10 | 4 | 0.9289 | 0.9141 | +0.0148 | well-fit |
| 400 | 0.05 | 6 | 0.9656 | 0.9304 | +0.0351 | well-fit |
| 600 | 0.05 | 8 | 0.9955 | 0.9444 | +0.0511 | overfit-ის ზღვარი |
| **800** | **0.03** | **10** | **0.9987** | **0.9475** | **+0.0513** | **CHOSEN** |
| 800 | 0.30 | 12 | 1.0000 | 0.9448 | +0.0552 | overfit demo |

**Gradient Boosting (HistGB)** (`GradientBoosting_Training`):

| learning_rate | max_iter | max_depth | train AUC | val AUC | gap | რეჟიმი |
|---|---|---|---|---|---|---|
| 0.50 | 50 | 3 | 0.8108 | 0.8147 | -0.0039 | underfit |
| 0.10 | 100 | 4 | 0.9124 | 0.9040 | +0.0084 | well-fit |
| 0.05 | 300 | 6 | 0.9518 | 0.9229 | +0.0289 | well-fit |
| **0.05** | **500** | **8** | **0.9753** | **0.9308** | **+0.0445** | **CHOSEN** |
| 0.01 | 800 | 10 | 0.9451 | 0.9207 | +0.0244 | well-fit |
| 0.50 | 800 | 12 | 0.8465 | 0.8286 | +0.0179 | LR ზედმეტად მაღალი → divergence |

**Random Forest** (`RandomForest_Training`):

| n_estimators | max_depth | min_samples_leaf | train AUC | val AUC | gap | რეჟიმი |
|---|---|---|---|---|---|---|
| 50 | 4 | — | 0.8313 | 0.8397 | -0.0084 | underfit |
| 100 | 8 | — | 0.8661 | 0.8680 | -0.0018 | well-fit |
| 200 | 12 | — | 0.9048 | 0.8859 | +0.0189 | well-fit |
| 300 | 16 | — | 0.9445 | 0.8992 | +0.0452 | overfit-ის ზღვარი |
| **200** | **None** | **20** | **0.9667** | **0.9094** | **+0.0572** | **CHOSEN** |
| 200 | None | 1 | 1.0000 | 0.9182 | +0.0818 | severe overfit |

**Decision Tree** (`DecisionTree_Training`):

| max_depth | min_samples_leaf | train AUC | val AUC | gap | რეჟიმი |
|---|---|---|---|---|---|
| 3 | — | 0.6896 | 0.6934 | -0.0037 | underfit |
| **6** | **—** | **0.8009** | **0.8032** | **-0.0023** | **CHOSEN (well-fit)** |
| 10 | — | 0.8309 | 0.7947 | +0.0362 | overfit-ის დასაწყისი |
| 16 | — | 0.8683 | 0.7937 | +0.0746 | overfit |
| None | 50 | 0.9552 | 0.8399 | +0.1153 | მაღალი val AUC, მაგრამ ტრენი 0.95 → ზედმეტად overfit, selection_score-ით ჩავარდა |
| None | 1 | 1.0000 | 0.7410 | +0.2590 | classic catastrophic overfit |

DT-ის შემთხვევაში selection_score-მა სწორად დახარჯა `min_samples_leaf=50` row-ი (val AUC 0.84 მაგრამ gap 0.12) და აირჩია უფრო ჯანმრთელი `max_depth=6` (val AUC 0.80, gap ~0).

**LogReg L1 / L2** (`LogisticRegression_L1_Training` / `_L2_Training`):

ცვლადი მხოლოდ `C` ∈ {0.001, 0.01, 0.1, 1.0, 10.0} — შესაბამისი 5 run თითოეულ ექსპერიმენტში. დაბალი `C` heavy regularisation = underfit, მაღალი `C` = სუსტი regularisation, შესაძლო overfit. ანალოგიური structure DagsHub-ის `LogisticRegression_L*_Training` ექსპერიმენტში ხილულია.

**AdaBoost** (`AdaBoost_Training`):

| n_estimators | learning_rate | base_depth | რეჟიმი |
|---|---|---|---|
| 50 | 1.0 | 1 | underfit (stumps) |
| 100 | 1.0 | 1 | underfit-ის ზღვარი |
| 200 | 0.5 | 3 | well-fit |
| 300 | 0.1 | 5 | well-fit |
| 500 | 0.5 | 8 | overfit demo (heavy base estimator) |

რეალური CV შედეგები DagsHub-ზე — ექსპერიმენტი დამთავრების შემდეგ ლოგდება იმავე სტრუქტურით.

შერჩეული combo შემდეგ გადის 3-fold StratifiedKFold-ს — სტრატიფიცირება აუცილებელია იმბალანსირებული fraud-ისთვის.

### საბოლოო Pipeline

- refit `pd.concat([X_train, X_val])`-ზე — val-ი HP-tuning-ში უკვე გამოვიყენე, deployed model-ი მაქსიმალური მონაცემიდან იღებს ცოდნას.
- `mlflow.sklearn.log_model(name='pipeline', signature=infer_signature(...), input_example=...)` — ხელახალი ჩატვირთვა მუშაობს raw DataFrame-ზე.
- `cache_architecture_result(...)` ანახლებს `results_cache.json`-ს.
- `register_if_better(final_run_id, cv_summary['cv_val_roc_auc_mean'])` — ადარებს ამ run-ს registry-ში არსებულ `IEEEFraudBestModel`-ს და მხოლოდ მაშინ ანაცვლებს, თუ უკეთესი CV ROC-AUC აქვს. ანუ ვინც ბოლო რანის შედეგად გაიმარჯვა, ის რჩება registry-ში.

### შედარების ცხრილი

`model_inference.ipynb`-ის ბოლო section-ი (`## 5. Architecture comparison`) კითხულობს `results_cache.json`-ს და ბეჭდავს leaderboard-ს, დახარისხებულს CV ROC-AUC-ით. ამავე უჯრედში registry-დან ჩამოვტვირთავთ `IEEEFraudBestModel`-ის უახლესი ვერსიის run-ს და ვადასტურებთ რომელი არქიტექტურაა მოქმედი ჩემპიონი.

DagsHub MLflow UI-ში იგივე შედარება ხილულია experiments-ის ჯვარედინი ხედვით, ხოლო registry-ის ვერსიების ისტორია გვიჩვენებს რომელი არქიტექტურა ცვლიდა წინამორბედს.

## MLflow Tracking

ბმული: https://dagshub.com/Saba0033/ML_Asgn2.mlflow/#/experiments

### სტრუქტურა

თითოეული არქიტექტურა — ცალკე ექსპერიმენტი, შიგნით კი დასახელებული run-ები pipeline-ის ეტაპების მიხედვით. მაგალითი XGBoost-ისთვის:

```
XGBoost_Training/
    XGBoost_Cleaning_fill-999
    XGBoost_Cleaning_fill0
    XGBoost_FeatureEngineering
    XGBoost_FeatureSelection_variance
    XGBoost_FeatureSelection_correlation
    XGBoost_FeatureSelection_model_based
    XGBoost_HP_n100_lr0p3_d3_ss1p0_cs1p0
    XGBoost_HP_n200_lr0p1_d4_ss0p9_cs0p9
    XGBoost_HP_n400_lr0p05_d6_ss0p8_cs0p8
    XGBoost_HP_n600_lr0p05_d8_ss0p8_cs0p7
    XGBoost_HP_n800_lr0p03_d10_ss0p7_cs0p6
    XGBoost_HP_n800_lr0p3_d12_ss1p0_cs1p0
    XGBoost_CrossValidation
    XGBoost_FinalPipeline
```

LogReg-ის HP run-ები: `LogReg_L2_HP_C0p001`, ... `LogReg_L2_HP_C10p0`.
LogReg-ის FS run-ები: `LogReg_L2_FeatureSelection_variance`, `_l1_logreg`, `_rf_importance`.

### ჩაწერილი მეტრიკები

თითოეული run ლოგავს train + val ბანდლს (`evaluate_train_val` `src/mlflow_utils.py`-ში):

| მეტრიკა | აღწერა |
|---|---|
| `train_roc_auc`, `val_roc_auc` | მთავარი მეტრიკა (Kaggle-ის შეფასება) |
| `train_pr_auc`, `val_pr_auc` | Precision-Recall AUC — მნიშვნელოვანია იმბალანსირებული მონაცემებისთვის |
| `train_accuracy`, `val_accuracy`, `train_f1`, `val_f1`, ... | სრულყოფისთვის |
| `overfit_gap` | `train_roc_auc - val_roc_auc` |
| `selection_score` | overfit-აჯარიმებული val AUC (HP runs-ში) |
| `n_features_kept` | feature selection runs-ში |
| `cv_val_roc_auc_mean / std`, `cv_val_pr_auc_mean`, `cv_overfit_gap` | CV runs-ში |

### საუკეთესო მოდელის შედეგები

- **Registry champion**: `IEEEFraudBestModel`, latest version
- **არქიტექტურა**: XGBoost
- **Best HP**: `n_estimators=800, learning_rate=0.03, max_depth=10, subsample=0.7, colsample_bytree=0.6`
- **CV ROC-AUC**: 0.9354 ± 0.0040 (3-fold StratifiedKFold, SAMPLE_FRAC=0.3)
- **CV PR-AUC**: 0.6968
- **Final val ROC-AUC** (refit on train+val): 0.9475
- **Cleaning**: numeric_fill = 0.0
- **Feature selector**: CorrelationPruner (threshold=0.95), 290 of 432 features kept

საბოლოო Pipeline ლოგდება `signature` + `input_example`-ით, ანუ Model Registry-დან ჩატვირთული მოდელი იცის რომელ სვეტებს ელოდება. `model_inference.ipynb` ოპერირებს raw test DataFrame-ზე, ყოველგვარი დამატებითი preprocessing-ის გარეშე — ყველა ეტაპი (engineer + preprocessor + selector + clf) Pipeline-ის შიგნით სრულდება.

ყოველი registered version-ი ავტომატურად ღებულობს tags + description-ს `register_if_better(...)`-ის წყალობით:

```
Architecture: XGBoost
Best HP: colsample_bytree=0.6, learning_rate=0.03, max_depth=10, n_estimators=800, subsample=0.7
CV ROC-AUC: 0.9354 +/- 0.0040
CV PR-AUC:  0.6968
CV overfit gap: 0.0642
Final val ROC-AUC: 0.9475
```

## როგორ გავუშვა

```bash
# 1. გარემო
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. მონაცემები
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && unzip -o ieee-fraud-detection.zip && cd ..

# 3. EDA
jupyter notebook 00_eda.ipynb

# 4. ცალ-ცალკე ყოველი მოდელის notebook
# (თითოეული თვითონ ლოგავს MLflow-ში და ავტომატურად რეგისტრირდება IEEEFraudBestModel-ად
# თუ მისი CV ROC-AUC უკეთესია, ვიდრე registry-ში არსებული ვერსიის)
jupyter notebook model_experiment_XGBoost.ipynb
# ... და დანარჩენი 6

# 5. inference + Kaggle submission
jupyter notebook model_inference.ipynb
kaggle competitions submit -c ieee-fraud-detection -f submissions/submission.csv -m "from registry"
```
