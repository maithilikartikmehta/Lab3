# %% read data
from statistics import LinearRegression
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()

# %% describe the dataframe
train.describe(include="all")

# %% SalePrice distribution
import seaborn as sns

sns.displot(
    data=train,
    x="SalePrice",
    hue="CentralAir",
)

# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
import seaborn as sns

sns.displot(
    data=train,
    x="SalePrice",
    hue="BldgType",
)

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood 
import matplotlib.pyplot as plt

plt.figure(figsize=(16,12))
ax = sns.boxplot(
    data=train,
    y="SalePrice",
    x="Neighborhood"
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45
)
ax.set_title("Awesome")

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    pred[pred<0] = 0
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")

# %% your solution to the regression problem

from sklearn.linear_model import LinearRegression 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

cat_cols=[
    "BldgType",
    "Neighborhood",
]

num_cols=[
    "1stFlrSF",
    "2ndFlrSF",
    "MasVnrArea",
    "TotRmsAbvGrd",
    "LotArea",
]
ct =ColumnTransformer(
    [
        ("ohe", OneHotEncoder(), cat_cols),
        ("imp", SimpleImputer(), num_cols),
    ],
    remainder="passthrough",
)

reg = LinearRegression()

selected_columns = cat_cols + num_cols
train_x = train[selected_columns]
train_y = train["SalePrice"]

train_x = ct.fit_transform(train_x)
reg.fit(train_x, train_y)
print("Training Set Performance")
print(evaluate(reg, train_x, train_y))

truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x = ct.transform(test_x)
print("Test Set Performance")
print(evaluate(reg, test_x, test_y))

# %%
