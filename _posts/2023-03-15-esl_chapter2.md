---
layout: default
title:  "Figure 2.9 - Elements of Statistical Learning"
date:   2023-03-15 23:59:08 -0700
categories: esl
description: "EPE of 1NN vs. OLS"
---

<head>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
<style>
.katex-display > .katex {
  display: inline-block;
  white-space: nowrap;
  max-width: 100%;
  overflow-x: scroll;
  text-align: initial;
}
.katex {
  font: normal 1.21em KaTeX_Main, Times New Roman, serif;
  line-height: 1.2;
  white-space: normal;
  text-indent: 0;
}
</style>
</head>

## Figure 2.9 Simulation

Here we're comparing the EPE of 1-nearest neighbor to least squares. That is, we're plotting 

$$\frac{EPE_{1nn}}{EPE_{OLS}}$$

for two seperate cases: 

1. $$f(x) = x_1$$
2. $$f(x) = \frac{1}{2}(x_1 + 1)^3$$

We assume an additive error model $$Y = f(X) + \epsilon$$, where $$X \sim U(-1, 1)$$ and $$\epsilon \sim N(0, 1)$$

### Initialize Variables


```python
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import tqdm
import pandas as pd

sns.set()

N = 500

# number of simulations used to calculate (approximate) EPE
num_simulations = 10000
```


```python
def fit_predict_clf(clf, X, Y, X0, Y0):
    clf.fit(X, Y)
    return (clf.predict(X0) - Y0)**2

def sim(dim, num_simulations):
    X_0 = np.zeros(dim).reshape(1, -1) # initialize test point of appropriate dimension
    Y_01 = np.random.normal(size=1)
    Y_02 = np.random.normal(size=1) + 0.5

    all_pe_1nn_Y1 = []
    all_pe_linear_Y1 = []
    all_pe_1nn_Y2 = []
    all_pe_linear_Y2 = []

    for _ in range(num_simulations):
        X_shape = (N, dim)
        X = np.random.uniform(low=-1, high=1, size=X_shape)
        epsilon = np.random.normal(size=500)

        Y_01 = np.random.normal(size=1)
        Y_02 = np.random.normal(size=1) + 0.5

        Y_1 = X[:,0] + epsilon
        Y_2 = 0.5 * (X[:, 0] + 1)**3 + epsilon

        clf_linear = LinearRegression()
        clf_1nn = KNeighborsRegressor(n_neighbors=1)

        pe_1nn_Y1 =  fit_predict_clf(clf_1nn, X, Y_1, X_0, Y_01)
        pe_linear_Y1 = fit_predict_clf(clf_linear, X, Y_1, X_0, Y_01)

        pe_1nn_Y2 =  fit_predict_clf(clf_1nn, X, Y_2, X_0, Y_02)
        pe_linear_Y2 = fit_predict_clf(clf_linear, X, Y_2, X_0, Y_02)

        all_pe_1nn_Y1.append(pe_1nn_Y1)
        all_pe_linear_Y1.append(pe_linear_Y1)

        all_pe_1nn_Y2.append(pe_1nn_Y2)
        all_pe_linear_Y2.append(pe_linear_Y2)


    # print(f'dim: {dim}')
    # print(f'EPE 1nn Y1: {np.mean(all_pe_1nn_Y1)}')
    # print(f'EPE OLS Y1: {np.mean(all_pe_linear_Y1)}')
    
    return np.mean(all_pe_1nn_Y1) / np.mean(all_pe_linear_Y1), np.mean(all_pe_1nn_Y2) / np.mean(all_pe_linear_Y2)
```


```python
epe_linear = []
epe_nonlinear = []

for dim in tqdm.trange(1, 10):
    epe_linear_ratio, epe_nonlinear_ratio = sim(dim, num_simulations)
    epe_linear.append(epe_linear_ratio)
    epe_nonlinear.append(epe_nonlinear_ratio)
```

    100%|██████████| 9/9 [07:25<00:00, 49.48s/it]



```python
cubic_df = pd.DataFrame({'EPE_ratio': epe_nonlinear, 'type': 'Cubic', 'idx': range(1, 10)})
linear_df = pd.DataFrame({'EPE_ratio': epe_linear, 'type': 'Linear', 'idx': range(1, 10)})
df = pd.concat([cubic_df, linear_df])
sns.lineplot(x='idx', y='EPE_ratio', hue='type', data=df, marker="o").set(title="Expected Predictiopn Error of 1NN vs. OLS")
```




    [Text(0.5, 1.0, 'Expected Prediction Error of 1NN vs. OLS')]




    
![png](/assets/images/esl_chapter2_files/esl_chapter2_5_1.png)
    

