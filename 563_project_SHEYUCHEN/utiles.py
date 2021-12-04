import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt


# data normalization
def data_normalization(df):
    # 中位数去极值
    dis_list = df.dtypes
    dis_list = dis_list[dis_list == 'object']
    df_tmp = df[set(df.columns).difference(set(dis_list.index))]
    if 'SalePrice' in df.columns:
        del df_tmp['SalePrice']
    df_max = df_tmp.mean() + 3 * (abs(df_tmp)).max()
    df_min = df_tmp.mean() - 3 * (abs(df_tmp)).max()
    df_tmp1 = (df_tmp - df_max > 0) * df_max
    df_tmp2 = (df_tmp - df_min < 0) * df_min
    df_tmp0 = (df_tmp1 + df_tmp2)
    df_tmp = (df_tmp - df_max < 0) * df_tmp
    df_tmp = (df_tmp - df_min > 0) * df_tmp
    df_tmp += df_tmp0
    # 标准化
    df_tmp = (df_tmp - df_tmp.mean()) / df_tmp.std()
    if 'SalePrice' in df.columns:
        df_tmp = pd.merge(df[['SalePrice']], df_tmp, left_index=True, right_index=True, how='left')
    return df_tmp


# discrete data impute
def data_impute(df, data_type='numeric', feature_y='SalePrice'):
    df_y = df[feature_y]
    df = df[df.columns.difference([feature_y])]
    for col_i in list(df.columns):
        if (pd.isna(df[col_i]) * 1).sum() == 0:
            continue
        if df[col_i].dtypes in ['int64', 'float64']:
            if data_type == 'character':
                continue
            feature_corr = df.corr(method='pearson')[col_i].sort_values(ascending=False)
            feature_corr = feature_corr.loc[feature_corr > 0.3].index.to_list()[1:]
            best_i = 0
            best_j = float('-inf')
            feature_corr_t = []
            feature_tmp = df.dropna(subset=[col_i])
            for i in range(len(feature_corr)):
                if (pd.isna(df[feature_corr[i]]) * 1).sum() > 0:
                    continue
                feature_corr_t.append(feature_corr[i])
                lr_feature = sm.OLS(feature_tmp[col_i], sm.add_constant(feature_tmp[feature_corr_t]))
                result = lr_feature.fit()
                if best_j < result.rsquared:
                    best_i = len(feature_corr_t)
                    beat_j = result.rsquared
                    # print('Parameters: ', result.params)
                    # print('Standard errors: ', result.bse)
                    # print('Predicted values: ', result .predict())
                    # print(result.summary())
            lr_feature = sm.OLS(feature_tmp[col_i], sm.add_constant(feature_tmp[feature_corr_t[:best_i]]))
            result = lr_feature.fit()
            df_tmp = df.loc[pd.isna(df[col_i]), feature_corr_t[:best_i]]
            df_tmp.insert(0, 'const', 1)
            df.loc[pd.isna(df[col_i]), col_i] = result.predict(df_tmp)
        else:
            if data_type == 'numeric':
                continue

            tmp = df[col_i].value_counts(dropna=True).index[0]
            df[col_i].fillna(tmp, inplace=True)
    return pd.concat([df, df_y], 1)


def greedy_target_encoding(df, feature_y='SalePrice', p_hyperparam=500):
    y = df[[feature_y]]
    for col_i in list(df.columns):
        if df[col_i].dtypes in ['int64', 'float64']:
            continue
        # elif col_i in ['GarageQual']:
        else:
            x = df[[col_i]]
            a_hyperparam = y.dropna().mean()[0]  # 是否删除向量中的缺失值
            df0 = pd.concat([x, y], axis=1)
            df_train = df0.dropna(subset=['SalePrice'])
            df1 = df_train.groupby(col_i).sum().reset_index()
            df2 = df_train.groupby(col_i).count().reset_index()
            df1.rename(columns={feature_y: 'tcfac'}, inplace=True)
            df2.rename(columns={feature_y: 'n'}, inplace=True)
            df1 = pd.merge(df1, df2, on=col_i, how='left')
            df1['target_encoding'] = (df1['tcfac'] + p_hyperparam * a_hyperparam) / (df1['n'] + p_hyperparam)
            df0 = pd.merge(df0.reset_index(), df1[[col_i, 'target_encoding']], on=col_i,
                           how='left').set_index('Id')[['target_encoding']]
            df0.rename(columns={'target_encoding': col_i}, inplace=True)
            del df[col_i]
            df = pd.merge(df, df0, left_index=True, right_index=True, how='left')
        # else:
        #     x = df[[col_i]]
        #     df0 = pd.get_dummies(x)
        #     pca = PCA(n_components=0.8)
        #     newX = pca.fit_transform(df0)
        #     # invX = pca.inverse_transform(df[feature_dict.values()])
        #     # pca.explained_variance_ratio_
        #     # df0 = pd.DataFrame(newX, columns=[col_i + '_dum_pca' + str(i) for i in range(len(newX.T))], index=df.index)
        #     # pca_df = pd.concat([pca_df, newX], 1)
        #     del df[col_i]
        #     df = pd.merge(df, df0, left_index=True, right_index=True, how='left')
    return df


# PCA 特征合成
def PCA_combine(df, feature_dict):
    pca_df = pd.DataFrame()
    for f_i in feature_dict.keys():
        f_j = feature_dict[f_i]
        pca = PCA(n_components=0.8)
        newX = pca.fit_transform(df[f_j])
        # invX = pca.inverse_transform(df[feature_dict.values()])
        # pca_explaination = pca.explained_variance_ratio_
        # plt.bar(range(6), pca_explaination)
        newX = pd.DataFrame(newX, columns=[f_i + '_pca' + str(i) for i in range(len(newX.T))], index=df.index)
        pca_df = pd.concat([pca_df, newX], 1)
    return pca_df


def lmse(y_pre, y):
    return np.sqrt(np.mean(np.square(np.log(y_pre) - np.log(y))))


def mse(y_pre, y):
    return np.sqrt(np.mean(np.square(y_pre - y)))


def selectX(df):
    # XGBClassifier
    df_y = df['SalePrice']
    df = df[df.columns.difference(['SalePrice'])]
    # xgb_reg = xgboost.XGBRegressor(random_state=0, booster='gbtree', reg_alpha=10, n_estimators=100, n_jobs=-1)
    # xgb_reg = xgb_reg.fit(df, df_y)
    # df_y_pred = xgb_reg.predict(df)
    # importances = xgb_reg.feature_importances_
    rnd_clf3 = RandomForestRegressor(random_state=0, n_estimators=100)
    rnd_clf3 = rnd_clf3.fit(df, df_y)
    df_y_pred = rnd_clf3.predict(df)
    importances = rnd_clf3.feature_importances_
    print(lmse(df_y_pred, np.array(df_y)))
    indices = np.argsort(importances)[::-1]
    importances_output = np.sort(importances)[::-1].cumsum()

    for i in range(0, len(importances_output)):
        if importances_output[i] > 0.96:
            break
    indices = indices[:i]

    importanceselected2 = []
    listselected2 = []
    for f in range(i):
        importanceselected2.append(importances[indices[f]])
        listselected2.append(df.columns[indices[f]])

    feature_importances_output = [importanceselected2, listselected2]
    feature_importances_output = pd.DataFrame(feature_importances_output,
                                              index=['featureSelected', 'listSelected'])
    print("Features sorte by their score:")

    return listselected2, feature_importances_output


def selectX_corr(df):
    y_train = df['SalePrice']
    X_train = df[df.columns.difference(['SalePrice'])]
    Xselected = X_train
    Xselected = Xselected.corr()
    Xselected[Xselected == 1] = 0
    Xcorr = Xselected[(Xselected > -0.7) & (Xselected < 0.7)]
    Xcorr = Xcorr.dropna(axis=0, how='any')
    Xcorr = Xcorr.index.to_list()
    a = len(Xcorr)
    Xcorr_temp = abs(Xselected.loc[:, Xselected.columns.difference(Xcorr)]).mean()
    Xcorr_temp = Xcorr_temp.sort_values()
    if len(Xcorr_temp.index) > 0:
        Xcorr.append(Xcorr_temp.index[0])
        while True:
            Xcorr_temp = Xselected.loc[Xcorr_temp.index[0], Xselected.columns.difference(Xcorr)]
            Xcorr_temp = abs(Xcorr_temp).sort_values()
            Xcorr.append(Xcorr_temp.index[0])
            if len(Xcorr) >= (a + (len(X_train.columns) - a) / 2):
                break

    return Xcorr


def select_lasso(df):
    df_y = df['SalePrice']
    df = df[df.columns.difference(['SalePrice'])]
    model = LassoCV()
    model.fit(df, df_y)

    # 训练后选择的lasso系数
    # print(model.alpha_)
    # 训练后线性模型参数
    params = model.coef_
    feature = df.columns[params != 0]
    df_y_pred = model.predict(df)
    print(lmse(df_y_pred, np.array(df_y)))
    return feature


def create_plt1(df, x_num=10):
    f, ax = plt.subplots(1, 1, figsize=(25, 10), dpi=200)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    xticks = [x for x in range(0, len(df.index), x_num)]
    xlabels = [df.index[x] for x in xticks]
    xticks.append(len(df.index))
    xlabels.append(df.index[-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=20, rotation=x_num)
    return f, ax


def create_plt2(df, x_num=10):
    f, ax1 = create_plt1(df)
    ax2 = ax1.twinx()
    xticks = [x for x in range(0, len(df.index), x_num)]
    xlabels = [df.index[x] for x in xticks]
    xticks.append(len(df.index))
    xlabels.append(df.index[-1])
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, fontsize=20, rotation=x_num)
    return f, ax1, ax2

