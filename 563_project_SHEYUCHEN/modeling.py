import numpy as np
import pandas as pd
import os
from utiles import data_impute, greedy_target_encoding, data_normalization, PCA_combine, selectX_corr, lmse, \
    select_lasso, mse, create_plt2
import xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# prepare data
data_path = '.\house-prices-advanced-regression-techniques'
train_data_path = os.path.join(data_path, 'train.csv')
test_data_path = os.path.join(data_path, 'test.csv')
# load data
train_data = pd.read_csv(train_data_path, index_col=0)
test_data = pd.read_csv(test_data_path, index_col=0)
test_data['SalePrice'] = np.nan
train_data['SalePrice'] = train_data['SalePrice']
data_raw = pd.concat([train_data, test_data])
# data_raw = utils.shuffle(data_raw)  # shuffle data
# data description
train_data_description = train_data.describe()
train_data_description.to_csv('./for_view/train_data_description.csv')
test_data_description = test_data.describe()
test_data_description.to_csv('./for_view/test_data_description.csv')

# # Impute MasVnrArea
# # cal corr of MasVnrArea
# MasVnrArea_corr = data_raw.corr(method='pearson')['MasVnrArea'].sort_values(ascending=False)
# # corr of MasVnrArea > 0.3
# MasVnrArea_corr = MasVnrArea_corr.loc[MasVnrArea_corr > 0.3].index.to_list()
# MasVnrArea_tmp = data_raw.dropna(subset=['MasVnrArea'])
# MasVnrArea_tmp = data_normalization(MasVnrArea_tmp)
# 'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea', 'GarageCars', 'YearBuilt', 'BsmtFinSF1'
# lr_MasVnrArea = sm.OLS(MasVnrArea_tmp['MasVnrArea'], sm.add_constant(MasVnrArea_tmp[['OverallQual', 'GrLivArea']]))
# sns.pairplot(MasVnrArea_tmp[['MasVnrArea', 'OverallQual', 'GrLivArea']])
# plt.show()
# # need to normalization**
# result = lr_MasVnrArea.fit()
# # # print('Parameters: ', result.params)
# # # print('Standard errors: ', result.bse)
# # # print('Predicted values: ', result .predict())
# # # print(result.summary())
# data_raw.loc[pd.isna(data_raw['MasVnrArea']), 'MasVnrArea'] = result.predict(sm.add_constant(data_raw.loc[pd.isna(
#     data_raw['MasVnrArea']), ['OverallQual', 'GrLivArea']]))
# 批量impute 原data的空值
df = data_impute(data_raw, data_type='character')
df = greedy_target_encoding(df)
# df0 = df['1stFlrSF']
df = data_normalization(df)
# df1 = df['1stFlrSF']
# df0 = pd.concat([df0, df1], 1)
# df0.columns = ['firstFlrSF', 'normailized_firstFlrSF']
# df00 = df0
# df0 = df0.iloc[:100, :]
# f, ax1, ax2 = create_plt2(df0)
# f1 = ax1.plot(df0.index, df0.iloc[:, [0]], 'blue', linewidth=1, label=df0.columns[0])
# f2 = ax2.plot(df0.index, df0.iloc[:, [1]], 'r--', linewidth=1, label=df0.columns[1])
# # 复合图例
# ax = f1 + f2
# labs = [x.get_label() for x in ax]
# plt.legend(ax, labs, loc='upper center', fontsize=25)
# ax1.set_xlabel('ID')
# ax1.set_ylabel(df0.columns[0])
# ax2.set_ylabel(df0.columns[1])
# plt.tight_layout()
# f.savefig('normalization_visualization.jpg')
# plt.close()
# sns.distplot(df00['firstFlrSF'])
# plt.title('firstFlrSF', fontsize=25)
# plt.savefig('firstFlrSF.jpg')
# plt.close()
# sns.distplot(df00['normailized_firstFlrSF'])
# plt.title('normailized_firstFlrSF', fontsize=25)
# plt.savefig('normalization_firstFlrSF.jpg')
df = data_impute(df, data_type='numeric')
df = data_normalization(df)
# PCA 将含义相近的features用PCA
# 面积相关和其他 LotArea	Utilities	1stFlrSF	2ndFlrSF	LowQualFinSF	GrLivArea	TotRmsAbvGrd	Functional	WoodDeckSF	Fence
feature_dict = {
    "住宅自身性质": ["MSSubClass", 'MSZoning', 'LotShape', 'LandContour', 'LandSlope', 'BldgType',
               'HouseStyle', 'YearRemodAdd', 'Foundation'],
    '住宅外部条件': ['LotFrontage', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
               'PavedDrive'],
    '房屋内外材质': ['OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 'RoofMatl', 'Exterior1st',
               'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond'],
    '地下室': ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'],
    '空调': ['Heating', 'HeatingQC', 'CentralAir', 'Electrical'],
    '房间': ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr'],
    '壁炉': ['Fireplaces', 'FireplaceQu'],
    '车库': ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond'],
    '门廊': ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'],
    '泳池': ['PoolArea', 'PoolQC'],
    '杂项': ['MiscFeature', 'MiscVal'],
    '销售': ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']
}
df_pca = PCA_combine(df, feature_dict)
df = pd.merge(df, df_pca, left_index=True, right_index=True, how='left')
# corr_selected = selectX_corr(df)
# corr_selected.append('SalePrice')
train_data = df.dropna(subset=['SalePrice'])
# test_data = df.loc[set(df.index).difference(train_data.index)]
test_data = df[pd.isna(df['SalePrice'])]
# feature_selected, feature_importance = selectX(train_data)
feature_selected = select_lasso(train_data)
train_data_y = np.log(train_data['SalePrice'])
train_data_x = train_data.copy()
del train_data_x['SalePrice']
test_data_x = test_data.copy()
del test_data_x['SalePrice']
test_data_x = test_data_x.sort_index(ascending=True)
X_train, X_validation, y_train, y_validation = train_test_split(train_data_x[feature_selected], train_data_y,
                                                                test_size=0.3, random_state=42)


xgb_reg = xgboost.XGBRegressor(random_state=0, booster='gblinear', n_estimators=500, n_jobs=-1)
xgb_reg = xgb_reg.fit(X_train[feature_selected], y_train)
df_train_y_pred = xgb_reg.predict(X_train[feature_selected])
print(mse(df_train_y_pred, np.array(y_train)))
df_validation_y_pred = xgb_reg.predict(X_validation[feature_selected])
print(mse(df_validation_y_pred, np.array(y_validation)))
xgb_reg = xgb_reg.fit(train_data_x[feature_selected], train_data_y)
df_train_y_pred = xgb_reg.predict(train_data_x[feature_selected])
# importances = xgb_reg.feature_importances_
# importances = pd.DataFrame(importances, index=feature_selected)
# df_train_y_pred = xgb_reg.predict(train_data_x[feature_selected])
# importances.sort_values(0, ascending=False, inplace=True)
# importances = importances.iloc[:30, :]
# plt.subplots(1, 1, figsize=(100, 25), dpi=500)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.bar(importances.index, importances.iloc[:, 0])
# plt.savefig('variance_importance.jpg')
print(mse(df_train_y_pred, np.array(train_data_y)))
df_test_y_pred = xgb_reg.predict(test_data_x[feature_selected])
df_test_y_pred = pd.DataFrame(np.exp(df_test_y_pred), columns=['SalePrice'], index=test_data_x.index)
df_test_y_pred.to_csv('submission.csv')