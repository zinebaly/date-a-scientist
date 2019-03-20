import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#change to project directory
os.chdir("/Users/zua23/Desktop/Machine_Learning/capstone_starter")

#Loading data:
df = pd.read_csv("profiles.csv")

# =============================================================================
# Data structure: 
#- body_type
#- diet
#- drinks
#- drugs
#- education
#- ethnicity
#- height
#- income
#- job
#- offspring
#- orientation
#- pets
#- religion
#- sex
#- sign
#- smokes
#- speaks
#- status
# =============================================================================
#Exploring data

# =============================================================================
# print("body_type \n" , df.body_type.head())
# print("diet \n" ,df.diet.head())
# print("drinks \n" ,df.drinks.head())
# print("drugs \n" ,df.drugs.head())
# print("education \n" ,df.education.head())
# print("ethnicity \n" ,df.ethnicity.head())
# print("height \n" ,df.height.head())
# print("income \n" ,df.income.head())
# print("job \n" ,df.job.head())
# print("offspring \n" ,df.offspring.head())
# print("orientation \n" ,df.orientation.head())
# print("pets \n" ,df.pets.head())
# print("religion \n" ,df.religion.head())
# print("sex \n" ,df.sex.head())
# print("sign \n" ,df.sign.head())
# print("smokes \n" ,df.smokes.head())
# print("speaks \n" ,df.speaks.head())
# print("status \n" ,df.status.head())
# =============================================================================

# Visualize some of the Data
#age
os.chdir("/Users/zua23/Desktop")
# =============================================================================
# fig1 = plt.figure()
# plt.hist(df.age, bins=20)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.xlim(16, 80)
# #plt.show()
# plt.savefig("age", dpi=200)
# 
# #height
# fig2 = plt.figure()
# plt.hist(df.height, bins=20)
# plt.xlabel("height")
# plt.ylabel("Frequency")
# plt.xlim(50, 80)
# #plt.show()
# plt.savefig("height", dpi=200)
# 
# #income
# fig3 = plt.figure()
# plt.hist(df.income, bins=20)
# plt.xlabel("income")
# plt.ylabel("Frequency")
# #plt.show()
# plt.savefig("income", dpi=200)
# =============================================================================


#print(df.sex.value_counts()) # there are 35829 males and 24117 females
#print(df.religion.value_counts())
#print(df.drinks.value_counts())
#print(df.drugs.value_counts())
#print(df.smokes.value_counts())
#print(df.diet.value_counts())
#print(df.education.value_counts())

# =============================================================================
# # Argumeting data
# =============================================================================
## drinks
all_data = df
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, 
                 "very often": 4, "desperately": 5}
all_data["drinks_code"] = all_data.drinks.map(drink_mapping)

## drugs
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
all_data["drugs_code"] = all_data.drugs.map(drugs_mapping)

## smokes
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, 
                  "trying to quit": 4}
all_data["smokes_code"] = all_data.smokes.map(smokes_mapping)

## diet
diet_mapping = {"mostly anything": 0, "anything": 1, "strictly anything": 2, 
                "mostly vegetarian": 3, "mostly other": 4, "strictly vegetarian": 5, 
                "vegetarian": 6, "strictly other": 7, "mostly vegan": 8, "other": 9,
                "strictly vegan": 10, "vegan": 11, "mostly kosher": 12,
                 "mostly halal": 13, "strictly halal": 14, "strictly kosher": 15,
                 "halal": 16, "kosher": 17}
all_data["diet_code"] = all_data.diet.map(diet_mapping)

## sex 
sex_mapping = {'m':0, 'f':1}
all_data["sex_code"] = all_data.sex.map(sex_mapping)

## religion
religion_mapping = {"agnosticism": 0, "other": 1, "agnosticism but not too serious about it": 2,
                    "agnosticism and laughing about it": 3, "catholicism but not too serious about it": 4, 
                    "atheism": 5, "other and laughing about it": 6, "atheism and laughing about it": 7,
                    "christianity": 8, "christianity but not too serious about it": 9,
                    "other but not too serious about it": 10, "judaism but not too serious about it": 11, 
                    "atheism but not too serious about it": 12, "catholicism": 13, "christianity and somewhat serious about it": 14,
                    "atheism and somewhat serious about it": 15, "other and somewhat serious about it": 16,
                    "catholicism and laughing about it": 17, "judaism and laughing about it": 18, "buddhism but not too serious about it": 19,
                    "agnosticism and somewhat serious about it": 20, "judaism": 21, "christianity and very serious about it": 22,
                    "atheism and very serious about it": 23, "catholicism and somewhat serious about it": 24, "other and very serious about it": 25, 
                    "buddhism and laughing about it": 26, "buddhism": 27, "christianity and laughing about it": 28, "buddhism and somewhat serious about it": 29,
                    "agnosticism and very serious about it": 30, "judaism and somewhat serious about it": 31, "hinduism but not too serious about it": 32, "hinduism": 33,
                    "catholicism and very serious about it": 34, "buddhism and very serious about it": 35, "hinduism and somewhat serious about it": 36, 
                    "islam": 37, "hinduism and laughing about it": 38, "islam but not too serious about it": 39, "judaism and very serious about it": 40,
                    "islam and somewhat serious about it": 41, "islam and laughing about it": 42, "hinduism and very serious about it": 43, "islam and very serious about it": 44 }
all_data["religion_code"] = all_data.religion.map(religion_mapping)

## combining the essays
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_data["essay_len"] = all_essays.apply(lambda x: len(x)) 

# =============================================================================
# ## Normalize your Data!
# =============================================================================
feature_data1 = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'diet_code', 'religion_code', 'sex_code', 'income']]
# let's remove NaN from all the columns: 
feature_data = feature_data1.dropna(subset = ['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'diet_code', 'religion_code', 'sex_code', 'income'])


x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

# =============================================================================
# ## Classification: Can we predict sex with diet, drinks_code, drugs code, 
#  smokes code and essay_len?
# =============================================================================
training_set, validation_set = train_test_split(feature_data, random_state = 100)
#support victor machine
classifier1 = SVC(kernel = 'rbf', gamma='scale') #kernel = 'linear'
classifier1.fit(training_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code' , 'essay_len']], training_set.sex_code)
gender_prediction1 = classifier1.predict(validation_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']])
#print(classifier1.score(validation_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']], validation_set.sex_code ))
# score of 0.6025113215314944
print(accuracy_score(validation_set.sex_code,gender_prediction1))
print(recall_score(validation_set.sex_code,gender_prediction1))
#print(precision_score(validation_set.sex_code,gender_prediction1))
#print(f1_score(validation_set.sex_code,gender_prediction1))


# Naive Bayes 
classifier2 = MultinomialNB()
classifier2.fit(training_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']], training_set.sex_code)
gender_prediction2 = classifier2.predict(validation_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']])
#print(classifier2.score(validation_set[['diet_code', 'drinks_code' , 'drugs_code', 'smokes_code', 'essay_len']], validation_set.sex_code))
# score of 0.6035405516673528
print(accuracy_score(validation_set.sex_code,gender_prediction2))
print(recall_score(validation_set.sex_code,gender_prediction2))
#print(precision_score(validation_set.sex_code,gender_prediction2))
#print(f1_score(validation_set.sex_code,gender_prediction2))


# k-Neighbors Classifier
accuracies = []
k_list = []
for k in range(100):
  classifier = KNeighborsClassifier(n_neighbors = k+1)
  classifier.fit(training_set[['diet_code', 'drinks_code', 'drugs_code', 'smokes_code', 'essay_len']], training_set.sex_code)
  score = classifier.score(validation_set[['diet_code', 'drinks_code' , 'drugs_code', 'smokes_code', 'essay_len']], validation_set.sex_code)
  k_list.append(k+1)
  accuracies.append(score)
  
fig4 = plt.figure()
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Gender prediction accuracy")
#plt.savefig("Gender_prediction_accuracy", dpi=200)

# =============================================================================
# Regression: Can we predict income with length of essays, and 'drinks_code',
# 'drugs_code', 'smokes_code' ?
# =============================================================================
X_train = training_set[['essay_len', 'drinks_code', 'drugs_code', 'smokes_code']]
y_train = training_set[['income']]
X_test = validation_set[['essay_len', 'drinks_code', 'drugs_code', 'smokes_code']]
y_test = validation_set[['income']]
#plt.scatter(X_train,y_train)

## trying a linear regressor
line_fitter = LinearRegression()
line_fitter.fit(X_train, y_train)
income_predict = line_fitter.predict(X_train)
#plt.plot(X_train, income_predict)
#plt.show()

# evaluation of the linear regression
R_square_train = line_fitter.score(X_train,y_train)
R_square_test = line_fitter.score(X_test,y_test)
print(R_square_train, R_square_test) # very poor fit! 
# 0.010191179585649301 0.00916541929409287

## trying K-Nearest Neighbors regressor
accuracies_regressor = []
k_neighbors = []
for k in range(100):
    regressor = KNeighborsRegressor(n_neighbors = k+1, weights = "distance") # n_neighbors = 5,
    regressor.fit(X_train,y_train)
    score = regressor.score(X_test,y_test)
    k_neighbors.append(k+1)
    accuracies_regressor.append(score)

fig5 = plt.figure()    
plt.plot(k_neighbors, accuracies_regressor)
plt.xlabel("k")
plt.ylabel("Validation accuracy of the regressor")
plt.title("Income prediction accuracy")
plt.savefig("Income_prediction_accuracy", dpi=200)
