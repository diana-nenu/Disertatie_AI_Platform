# Training Log - Sesiunea 2 Spania pe Databricks

Generat: 2026-06-15T17:20:31.633300
Mod rulare: **full**
Cluster: UTM Databricks (16.4 LTS)
Output dir: /tmp/disertatie_ai_platform

## 1. Tabel comparativ modele
```
    dataset   platform                model     rmse      mae       r2     mape
pret_spania databricks XGBoost_tuned_Optuna 2.001339 1.485086 0.969564 2.484066
pret_spania databricks         RandomForest 2.132082 1.584549 0.965458 2.611547
pret_spania databricks              XGBoost 2.238229 1.713987 0.961933 2.826423
pret_spania databricks                Lasso 2.395008 1.701336 0.956413 2.931524
pret_spania databricks                Ridge 2.625041 2.045881 0.947638 3.461776
pret_spania databricks     LinearRegression 2.641757 2.065290 0.946969 3.488959
pret_spania databricks                 LSTM 4.716024 3.881446 0.836071 7.073903
```

**Castigator**: XGBoost_tuned_Optuna
**RMSE castigator**: 2.00
**R^2 castigator**: 0.9696

## 2. Optuna best params
```python
{'n_estimators': 495, 'max_depth': 6, 'learning_rate': 0.04093139181661603, 'subsample': 0.9362591393769683, 'colsample_bytree': 0.866898244191454}
```
Best CV RMSE: 2.71
Trials completate: 28, pruned: 22

## 3. Feature importance top 30 (XGBoost tunat)
```
                   feature  importance
        price actual_lag_1    0.477354
  price actual_roll_mean_3    0.365426
       price actual_lag_24    0.020231
           price day ahead    0.018834
                  hour_cos    0.018135
        price actual_lag_2    0.016927
      price actual_lag_168    0.013276
                  hour_sin    0.005907
                       day    0.005144
                      year    0.003845
                      hour    0.002922
                is_weekend    0.002431
                 dayofweek    0.002300
                    season    0.002276
                 month_cos    0.002223
 price actual_roll_mean_24    0.002200
 price actual_roll_std_168    0.001989
   price actual_roll_std_3    0.001693
price actual_roll_mean_168    0.001611
                weekofyear    0.001560
                 dayofyear    0.001522
  price actual_roll_std_24    0.001497
                   quarter    0.001461
      virtual_hour_of_year    0.001404
       virtual_day_of_year    0.001329
                 month_sin    0.001257
                     month    0.001249
        price actual_lag_3    0.001037
                wind_speed    0.001025
                   dow_sin    0.001016
```

## 4. SHAP top 20 (mean_abs_shap + directie)
```
                                   feature  mean_abs_shap  mean_shap
                        price actual_lag_1       9.043818   6.612383
                           price day ahead       2.005707   1.547909
                  price actual_roll_mean_3       1.265365   0.940440
                                  hour_cos       0.901710  -0.098124
                      price actual_lag_168       0.648533   0.458455
                       price actual_lag_24       0.423449   0.245381
                                  hour_sin       0.422775   0.018225
                                       day       0.273500   0.051408
                        price actual_lag_2       0.244461  -0.064280
                                      hour       0.225364   0.014864
                 price actual_roll_mean_24       0.162001   0.121479
                                 dayofweek       0.124928  -0.005605
                price actual_roll_mean_168       0.105071   0.030183
               generation fossil hard coal       0.097304   0.049214
                        generation biomass       0.092292  -0.088300
                generation other renewable       0.085816  -0.050959
generation hydro run-of-river and poundage       0.072186   0.056700
                        price actual_lag_3       0.070668   0.052140
                 price actual_roll_std_168       0.069835  -0.023135
                       total load forecast       0.069305   0.004708
```
Expected value (predictia medie): 55.32 EUR/MWh

## 5. Lasso - features non-zero
Features non-zero: 19/78

## 6. LSTM training history
Epochs antrenate: 12
```
Epoch    Loss        Val Loss
1        0.2017      0.2218      
2        0.1005      0.1840      
3        0.0776      0.1600      
4        0.0647      0.1486      
5        0.0587      0.1478      
6        0.0519      0.1427      
7        0.0478      0.1371      
8        0.0453      0.1434      
9        0.0418      0.1426      
10       0.0407      0.1464      
11       0.0384      0.1457      
12       0.0372      0.1486      
```

## 7. Setari utilizate
```python
N_TRAIN_LSTM: 20000
N_VAL_LSTM: 3000
N_TEST_LSTM: 4000
LSTM_UNITS: 64
LSTM_EPOCHS: 30
LSTM_BATCH: 128
OPTUNA_TRIALS: 50
OPTUNA_SPLITS: 3
N_TUNE: 20000
SHAP_SAMPLES: 2000
N_CV: 20000
CV_SPLITS: 5
RF_ESTIMATORS: 200
RF_DEPTH: None
XGB_ESTIMATORS: 300
XGB_DEPTH: 6
```

## 8. Date despre dataset
- Total randuri: 34896
- Numar features: 78
- Train: 24428, Val: 3489, Test: 6979