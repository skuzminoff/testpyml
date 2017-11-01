# coding=utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

pd.set_option('display.width', 256)


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

all_data = pd.concat([train_data, test_data])

print("----survived by class and sex")
print(train_data.groupby(["Pclass", "Sex"])["Survived"].value_counts(normalize=True))

describe_fields = ["Age", "Fare", "Pclass", "SibSp", "Parch"]

print("---- train: males")
print(train_data[train_data["Sex"] == "male"][describe_fields].describe())

print("---- test: males")
print(test_data[test_data["Sex"] == "male"][describe_fields].describe())

print("---- train: females")
print(train_data[train_data["Sex"] == "female"][describe_fields].describe())

print("---- test: females")
print(test_data[test_data["Sex"] == "female"][describe_fields].describe())

class DataDigest:
    def __init__(self):
        self.ages = None
        self.fares = None
        self.titles = None
        self.cabins = None
        self.families = None
        self.tickets = None

def get_title(name) :
    if pd.isnull(name):
        return "Null"

    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1).lower()
    else:
        return "None"

def get_family(row):
    last_name = row["Name"].split(",")[0]
    if last_name :
        family_size = 1 + row["Parch"] + row["SibSp"]
        if family_size > 3 :
            return "{0}_{1}".format(last_name.lower(), family_size)
        else:
            return "nofamily"
    else:
        return "unknown"


data_digest = DataDigest()
data_digest.ages = all_data.groupby("Sex")["Age"].median()
data_digest.fares = all_data.groupby("Pclass")["Fare"].median()

titles_trn = pd.Index(train_data["Name"].apply(get_title).unique())
titles_tst = pd.Index(test_data["Name"].apply(get_title).unique())
data_digest.titles = titles_tst

families_trn = pd.Index(train_data.apply(get_family, axis=1).unique())
families_tst = pd.Index(test_data.apply(get_family, axis=1).unique())
data_digest.families = families_tst

cabins_trn = pd.Index(train_data["Cabin"].fillna("unknown").unique())
cabins_tst = pd.Index(test_data["Cabin"].fillna("unknown").unique())
data_digest.cabins = cabins_tst

tickets_trn = pd.Index(train_data["Ticket"].fillna("unknown").unique())
tickets_tst = pd.Index(test_data["Ticket"].fillna("unknown").unique())
data_digest.tickets = tickets_tst


def get_index(item, index):
    if pd.isnull(item):
        return -1
    try:
        return index.get_loc(item)
    except KeyError:
        return -1

def munge_data(data, digest):
    #Age - замена пропусков на медиану в зав-ти от пола
    data["AgeF"] = data.apply(lambda r: digest.ages[r["Sex"]] if pd.isnull(r["Age"]) else r["Age"], axis=1)

    #Fare - замена пропусков на медиану в зав-ти от класса
    data["FareF"] = data.apply(lambda r: digest.fares[r["Pclass"]] if pd.isnull(r["Fare"]) else r["Fare"], axis=1)

    #Gender - замена
    genders = {"male" : 1, "female": 0}
    data["SexF"] = data["Sex"].apply(lambda s : genders.get(s))

    #Gender - расширение
    gender_dummies = pd.get_dummies(data["Sex"], prefix="SexD", dummy_na=False)
    data = pd.concat([data, gender_dummies], axis=1)

    #Embarkment - замена
    embarkments = {"U": 0, "S" : 1, "C": 2, "Q": 3}
    data["EmbarkedF"] = data["Embarked"].fillna("U").apply(lambda e: embarkments.get(e))

    #Embarkment - расширение
    embarkment_dummies = pd.get_dummies(data["Embarked"], prefix="EmbarkedD", dummy_na=False)
    data = pd.concat([data, embarkment_dummies], axis=1)

    #количество родственников на борту
    data["RelativesF"] = data["Parch"] + data["SibSp"]

    #человек-одиночка
    data["SingleF"] = data["RelativesF"].apply(lambda r: 1 if r == 0 else 0)

    #Deck - замена
    decks = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    data["DeckF"] = data["Cabin"].fillna("U").apply(lambda c: decks.get(c[0], -1))

    #Deck - расширение
    deck_dummies = pd.get_dummies(data["Cabin"].fillna("U").apply(lambda c: c[0]), prefix="DeckD", dummy_na=False)
    data = pd.concat([data, deck_dummies], axis=1)

    #Titles - расширение
    title_dummies = pd.get_dummies(data["Name"].apply(lambda n: get_title(n)),prefix="TitleD", dummy_na=False)
    data = pd.concat([data, title_dummies], axis=1)

    #замена текстов на индекс из соотв справочника или -1 если значение в справочнике
    #отсутствует (расширять не будем)
    data["CabinF"] = data["Cabin"].fillna("unknown").apply(lambda c: get_index(c, digest.cabins))

    data["TitleF"] = data["Name"].apply(lambda n: get_index(get_title(n), digest.titles))

    data["TicketF"] = data["Ticket"].apply(lambda t: get_index(t, digest.tickets))

    data["FamilyF"] = data.apply(lambda r: get_index(get_family(r), digest.families), axis=1)

    #для статистики
    age_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
    data["AgeR"] = pd.cut(data["Age"].fillna(-1), bins=age_bins).astype(object)

    return data;

train_data_munged = munge_data(train_data, data_digest)
test_data_munged = munge_data(test_data, data_digest)
all_data_munged = pd.concat([train_data_munged, test_data_munged])


predictors = [
              "Pclass",
              "AgeF",
              #"TitleF",
              #"TitleD_mr","TitleD_mrs", "TitleD_miss", "TitleD_master", "TitleD_ms", "TitleD_col", "TitleD_rev", "TitleD_dr",
              #"CabinF",
              "DeckF",
              #"DeckD_U", "DeckD_A", "DeckD_B", "DeckD_C", "DeckD_D", "DeckD_E", "DeckD_F", "DeckD_G",
              "FamilyF",
              "TicketF",
              "SexF",
              #"SexD_male", "SexD_female",
              #"EmbarkedF",
              #"EmbarkedD_S", "EmbarkedD_C", "EmbarkedD_Q",
              "FareF",
              "SibSp", "Parch",
              "RelativesF",
              "SingleF"]


#print("----- all data munged")
#print(all_data_munged.describe())
all_data_munged.to_csv("all_data_munged.csv")

scaler = StandardScaler()
scaler.fit(all_data_munged[predictors])

train_data_scaled = scaler.transform(train_data_munged[predictors])
test_data_scaled = scaler.transform(test_data_munged[predictors])

print("--- survived by age")
print(train_data_munged.groupby(["AgeR"])["Survived"].value_counts(normalize=True))

print("--- survived by gender and age")
print(train_data_munged.groupby(["Sex", "AgeR"])["Survived"].value_counts(normalize=True))

print("--- survived by class and age")
print(train_data_munged.groupby(["Pclass", "AgeR"])["Survived"].value_counts(normalize=True))

sns.set_palette("pastel")
sns.set_context("notebook", font_scale=1.1, rc={"lines.linewidth": 3.5})
sns.set_style("darkgrid")
sns.pairplot(train_data_munged, vars=["AgeF", "Pclass", "SexF"], hue="Survived", dropna=True)
plt.show()




    
    
            
                                 

