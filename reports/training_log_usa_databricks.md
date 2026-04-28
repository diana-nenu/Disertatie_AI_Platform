# Training Log - Sesiunea 1 USA pe Databricks

Generat: 2026-04-28T05:43:16.592454
Mod rulare: **full**
Cluster: UTM Databricks (16.4 LTS)
Output dir: /tmp/disertatie_ai_platform

## 1. Tabel comparativ modele
```
   dataset   platform            model        rmse         mae        r2      mape
consum_usa databricks    XGBoost_tuned  324.891408  235.980179  0.997495  0.752523
consum_usa databricks          XGBoost  328.050641  237.853317  0.997446  0.762749
consum_usa databricks     RandomForest  358.610681  253.321839  0.996948  0.813295
consum_usa databricks             LSTM  407.219435  300.335561  0.995808  0.980166
consum_usa databricks LinearRegression  514.832884  382.550198  0.993710  1.222194
consum_usa databricks          Prophet 7869.670262 6639.328291 -0.469717 21.113696
```

**Castigator**: XGBoost_tuned
**RMSE castigator**: 324.89
**R^2 castigator**: 0.9975

## 2. GridSearchCV best params
```python
{'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 300}
```

Best CV RMSE: 391.24

## 3. Feature importance top 30
```
              feature  importance
        PJME_MW_lag_1    0.901346
             hour_cos    0.051105
        PJME_MW_lag_2    0.008788
                 hour    0.006724
            month_cos    0.005389
             hour_sin    0.005307
        PJME_MW_lag_3    0.003507
 PJME_MW_roll_mean_24    0.003500
            dayofweek    0.002850
   PJME_MW_roll_std_3    0.002519
           weekofyear    0.000976
            month_sin    0.000949
       PJME_MW_lag_24    0.000923
 virtual_hour_of_year    0.000761
  virtual_day_of_year    0.000699
 PJME_MW_roll_std_168    0.000614
  PJME_MW_roll_std_24    0.000609
              dow_cos    0.000563
            dayofyear    0.000554
      PJME_MW_lag_168    0.000450
               season    0.000425
PJME_MW_roll_mean_168    0.000349
                month    0.000328
              dow_sin    0.000272
  PJME_MW_roll_mean_3    0.000218
                 year    0.000153
                  day    0.000121
           is_weekend    0.000000
              quarter    0.000000
```

## 4. LSTM training history
Epochs antrenate: 30
```
Epoch    Loss        Val Loss    MAE
1        0.1114      0.0383      0.2398      
2        0.0352      0.0236      0.1447      
3        0.0255      0.0185      0.1222      
4        0.0218      0.0153      0.1119      
5        0.0192      0.0136      0.1046      
6        0.0171      0.0110      0.0983      
7        0.0159      0.0110      0.0941      
8        0.0150      0.0092      0.0913      
9        0.0141      0.0084      0.0881      
10       0.0135      0.0078      0.0864      
11       0.0127      0.0080      0.0835      
12       0.0124      0.0074      0.0821      
13       0.0120      0.0073      0.0806      
14       0.0118      0.0068      0.0797      
15       0.0113      0.0060      0.0784      
16       0.0112      0.0059      0.0773      
17       0.0110      0.0063      0.0766      
18       0.0108      0.0053      0.0757      
19       0.0104      0.0055      0.0744      
20       0.0102      0.0053      0.0739      
21       0.0105      0.0060      0.0748      
22       0.0101      0.0053      0.0733      
23       0.0100      0.0052      0.0731      
24       0.0098      0.0050      0.0720      
25       0.0098      0.0049      0.0721      
26       0.0098      0.0047      0.0717      
27       0.0096      0.0044      0.0712      
28       0.0096      0.0052      0.0711      
29       0.0096      0.0048      0.0712      
30       0.0094      0.0047      0.0705      
```

## 5. Setari utilizate
```python
N_TRAIN_LSTM: 50000
N_VAL_LSTM: 5000
N_TEST_LSTM: 10000
LSTM_UNITS: 64
LSTM_EPOCHS: 30
LSTM_BATCH: 128
PROPHET_TRAIN_HOURS: 45000
N_TUNE: 30000
GRID_SPLITS: 3
N_CV: 50000
CV_SPLITS: 5
RF_ESTIMATORS: 200
RF_DEPTH: None
XGB_ESTIMATORS: 300
XGB_DEPTH: 6
```

## 6. Date despre dataset
- Total randuri: 145194
- Numar features: 29
- Train: 101637
- Val: 14519
- Test: 29038