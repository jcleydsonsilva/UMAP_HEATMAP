#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif


feature_df = pd.read_csv(sys.argv[1], sep="\t")

y = list(feature_df["go_id"])
X = feature_df.iloc[:,1:-2]

res = mutual_info_classif(X,y)

feat = list(X.columns)
out = open(f"{sys.argv[1]}_MI-Scrore.csv")
for i in range(0,len(res)):
    out.write (f"{feat[i]},{res[i]}")
out.close()
