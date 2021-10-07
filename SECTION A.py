import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv(r"C:\train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv(r"C:\test_Y3wMUE5_7gLdaTN.csv")

test.head()
train.columns
test.columns
train.dtypes

print("Shape of train dataset: ", train.shape)
train.head()

print('Shape of test dataset', test.shape)
test.head()

train["Loan"].count()
train['Loan'].value_counts(normalize=True) * 100

train["Gender"].count()
train['Gender'].value_counts()
(train['Gender'].value_counts(normalize=True) * 100)

print("1 a) Find out the number of male and female in loan applicants data.")

train['Gender'].value_counts(normalize=True).plot.bar(title='Gender of loan applicants data')
print(train['Gender'].value_counts(normalize=True) * 100)
plt.xlabel('Gender')
plt.ylabel('Number of loan applicants')
plt.show()

print(train["Married"].count())
print(train["Married"].value_counts())
train['Married'].value_counts(normalize=True) * 100

print("1 b) Find out the number of married and unmarried loan applicants.")

train['Married'].value_counts(normalize=True).plot.bar(title='Married Status of an applicant')

print('Yes-> Married and No-> Unmarried')

print(train['Married'].value_counts(normalize=True) * 100)
plt.xlabel('Marital_Status')
plt.ylabel('Number of loan applicants')
plt.show()

train["Self_Employed"].count()
train['Self_Employed'].value_counts()
train['Self_Employed'].value_counts(normalize=True) * 100

print("1 c) Find out the overall dependent status in the dataset.")

train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Dependent_Status')
print(train['Self_Employed'].value_counts(normalize=True) * 100)
plt.xlabel('Dependent_Status')
plt.ylabel('Number of loan applicants')
plt.show()

train["Education"].count()
train["Education"].value_counts()
train["Education"].value_counts(normalize=True) * 100

print("1 d) Find the count how many loan applicants are graduate and non graduate.")

train["Education"].value_counts(normalize=True).plot.bar(title="Education")
print(train["Education"].value_counts(normalize=True) * 100)
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

train["Property_Area"].count()
train["Property_Area"].value_counts()
train["Property_Area"].value_counts(normalize=True) * 100

print("1 e) Find out the count how many loan applicants property lies in urban, rural and semi-urban areas.")

train["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")
print(train["Property_Area"].value_counts(normalize=True) * 100)

plt.ylabel('Percentage')
plt.show()

train["Credit_History"].count()
train["Credit_History"].value_counts()
train['Credit_History'].value_counts(normalize=True) * 100

train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
print(train['Credit_History'].value_counts(normalize=True) * 100)
plt.xlabel('Debt')
plt.ylabel('Percentage')
plt.show()

print("3")
print(
    "To visualize and plot the distribution plot of all numerical attributes of the given train dataset i.e. "
    "ApplicantIncome,  CoApplicantIncome and LoanAmount.     ")

print("Applicant_Income distribution: ")
print(train["Applicant_Income"])
plt.figure(1)
plt.subplot(121)
sns.distplot(train["Applicant_Income"]);

plt.subplot(122)
train["Applicant_Income"].plot.box(figsize=(16, 5))
plt.show()

train.boxplot(column='Applicant_Income', by="Education")
plt.suptitle(" ")
plt.show()

print("Co-applicant Income distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(train["Co-applicantIncome"]);

plt.subplot(122)
train["Co-applicantIncome"].plot.box(figsize=(16, 5))
plt.show()

print("Loan Amount Variable")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df['Loan_Amount']);

plt.subplot(122)
train['Loan_Amount'].plot.box(figsize=(16, 5))

plt.show()
print("Loan Amount Term Distribution")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df["Loan_Amount_Term"]);

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16, 5))
plt.show()
