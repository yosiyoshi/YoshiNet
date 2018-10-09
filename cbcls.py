# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:37:08 2018

Modified from catboost tutorial code from:
https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/
@author: YosiYoshi
"""
from catboost import CatBoostClassifier
cat_features = [1,0,2]
train_data=[["a","b",1,4,5,6],["a","b",4,5,6,7],["c","d",30,40,50,60]]
train_labels=[1,-1,1]
test_data=[["a","b",2,4,6,8],["a","b",1,4,50,60],["c","d",40,80,160,320]]
model=CatBoostClassifier(iterations=2,learning_rate=1,depth=2)
model.fit(train_data,train_labels,cat_features)
preds_class=model.predict(test_data)
preds_proba=model.predict_proba(test_data)
preds_raw=model.predict(test_data,prediction_type="RawFormulaVal")