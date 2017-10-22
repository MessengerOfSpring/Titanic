import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale

data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")
print('data import complete.')

data_train['Sex'] = data_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
data_test['Sex'] = data_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
data_train['Cabin'] = data_train['Cabin'].apply(lambda x: 1 if x is not np.nan else 0)
data_test['Cabin'] = data_test['Cabin'].apply(lambda x: 1 if x is not np.nan else 0)


# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

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
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def pre_process(df):
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, dummies_Pclass, dummies_Embarked], axis=1)
    df = DataFrame(df)
    df['Age_scaled'] = scale(df['Age'])
    df['Fare_scaled'] = scale(df['Fare'])

    df.drop(['Pclass', 'Embarked', 'Name', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)

    return df


data_train, rfr = set_missing_ages(data_train)
data_train = pre_process(data_train)

data_test.loc[data_test['Fare'].isnull(), 'Fare'] = data_test['Fare'].mean()
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = pre_process(data_test)

print('processing OK.')

from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

test_df = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test_df)

result = DataFrame({
    'PassengerId': data_test['PassengerId'],
    'Survived': predictions.astype(np.int32)
})
result.to_csv('./data/logistic_prediction.csv', index=False)
