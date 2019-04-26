# -*- coding:UTF-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from keras import optimizers
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, BatchNormalization
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import sys
import io
import random


def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # model 1  gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    
    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # two models merge
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
 
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
 
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
 
    # y即目标年龄
    y = known_age[:, 0]
 
    # X即特征属性值
    X = known_age[:, 1:]
 
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
 
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
 
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
 
    return df, rfr

# 建立PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'


def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def feature_Engineering(combined_train_test):
    #“Embarked”项的缺失值不多，以众数来填充
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    # 为了后面的特征分析，这里我们将 Embarked 特征进行facrorizing
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    '''
    # 使用 pd.get_dummies 获取one-hot 编码
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
    '''
    
    # 为了后面的特征分析，这里我们也将 Sex 特征进行facrorizing
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    '''
    sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
    combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)
    '''
    
    # what is each person's title? 
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
    # 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing
    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    '''
    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
    combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
    combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)
    '''
    #一二三等舱各自的均价来填充
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
    #将团体票的票价分配到每个人的头上
    combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
    #使用binning给票价分等级
    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
    #对于5个等级的票价我们也可以继续使用dummy为票价等级分列
    combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
    '''
    fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
    combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
    '''
    combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)
    
    #分出每等舱里的高价和低价位        
    Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
    Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
    Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]
    # 建立Pclass_Fare Category
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
    pclass_level = LabelEncoder()
    # 给每一项添加标签
    pclass_level.fit(np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
    # 转换成数值
    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
    # dummy 转换
    '''
    pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
    combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
    '''
    #将 Pclass 特征factorize化
    combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

    #亲友的数量没有或者太多会影响到Survived
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])
    '''
    family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'], prefix=combined_train_test[['Family_Size_Category']].columns[0])
    combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)
    '''
    
    #利用融合模型预测的结果填充Age的缺失值
    #missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])
    missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
    combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)

    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
    # 将 Ticket_Letter factorize
    combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]
    # 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。
    # combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    # combined_train_test['Ticket_Number'].fillna(0, inplace=True)

    #replace missing value with U0
    combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
    
    '''
    Correlation = pd.DataFrame(combined_train_test[
        ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, linecolor='white', annot=True)
    plt.show()
    '''
    
    # StandardScaler will subtract the mean from each value then scale to the unit variance
    '''
    scaler = preprocessing.StandardScaler()
    combined_train_test['Age_scaled'] = scaler.fit_transform(combined_train_test['Age'].values.reshape(-1, 1))
    combined_train_test['Fare_scaled'] = scaler.fit_transform(combined_train_test['Fare'].values.reshape(-1, 1))
    '''
    
    #弃掉无用特征
    '''
    combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 
                          'Parch', 'SibSp', 'Family_Size_Category', 'Ticket', 'Name_length', 'Ticket_Letter'], axis=1,inplace=True)
    '''
    combined_train_test.drop(['PassengerId', 'Name', 'Parch', 'SibSp',  'Ticket',  'Ticket_Letter'], axis=1,inplace=True)


    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]
    
    titanic_train_data_X = train_data.drop(['Survived'],axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'],axis=1)
    
    return np.array(titanic_train_data_X), np.array(titanic_train_data_Y), np.array(titanic_test_data_X)



if __name__ == '__main__':
    train_df_org = pd.read_csv('./train.csv')
    test_df_org = pd.read_csv('./test.csv')
    test_Y_org = pd.read_csv('./submission.csv')
    test_df_org['Survived'] = 0
    combined_train_test = train_df_org.append(test_df_org)
    PassengerId = test_df_org['PassengerId']
    titanic_test_data_Y = test_Y_org['Survived']
    
    #特征工程处理
    titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X = feature_Engineering(combined_train_test)
    
    # build the model: a single LSTM
    print('Build model...')
    optimizer = optimizers.Adam(lr=0.001)
    model = Sequential()
    model.add(BatchNormalization(input_shape = (len(titanic_train_data_X[0]),)))
    model.add(Dense(32, activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='selu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(titanic_train_data_X, titanic_train_data_Y, batch_size=32, epochs=200)

    print(model.evaluate(titanic_train_data_X, titanic_train_data_Y))
    print(model.evaluate(titanic_test_data_X, titanic_test_data_Y))

    titanic_test_data_Y = model.predict_classes(titanic_test_data_X)
    pd.DataFrame({"PassengerId": PassengerId, "Survived": titanic_test_data_Y.reshape(-1)}).to_csv('titanic_submission.csv', index=False, header=True)
    
    print()