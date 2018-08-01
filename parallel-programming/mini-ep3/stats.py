import pandas as pd
import statsmodels.formula.api as sm

df = pd.read_csv('dataset-21.csv')

regression = sm.ols(formula="avg_time ~ vector_size + num_threads + n_ifs + vector_size * num_threads + vector_size * n_ifs + n_ifs * num_threads", data=df).fit()
print(regression.summary())