import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
train = pd.read_csv(r"C:\train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv(r"C:\test_Y3wMUE5_7gLdaTN.csv")

test.head()
print(train.columns)
print(test.columns)
print(train.dtypes)

print("Train dataset shape ", train.shape)
train.head()

print('Test dataset shape', test.shape)
test.head()

train["Loan_Status"].count()
train['Loan_Status'].value_counts(normalize=True) * 100

train['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan Status')
plt.ylabel('Loan Applicants ')
plt.xlabel('Status')
plt.show()

train["Gender"].count()
train['Gender'].value_counts()
(train['Gender'].value_counts(normalize=True) * 100)

print(" ")
print("1(a): Find out the number of male and female in loan applicants data.")
print(" ")
train['Gender'].value_counts(normalize=True).plot.bar(title='Gender of loan applicant data')
print(train['Gender'].value_counts(normalize=True) * 100)
plt.xlabel('Gender')
plt.ylabel('Number of loan applicant')
plt.show()

print(train["Married"].count())
print(train["Married"].value_counts())
train['Married'].value_counts(normalize=True) * 100

print("1(b) Find out the number of married and unmarried loan applicants.")
print(" ")
train['Married'].value_counts(normalize=True).plot.bar(title='Married Status of an applicant')
print('yes means married and no means unmarried')
print(train['Married'].value_counts(normalize=True) * 100)
plt.xlabel('Married Status')
plt.ylabel('Number of loan applicant')
plt.show()

train["Self_Employed"].count()
train['Self_Employed'].value_counts()
train['Self_Employed'].value_counts(normalize=True) * 100

print("1(c) Find out the overall dependent status in the dataset.")
print(" ")
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Dependent Status')
print(train['Self_Employed'].value_counts(normalize=True) * 100)
plt.xlabel('Dependent Status')
plt.ylabel('Number of loan applicant')
plt.show()

train["Education"].count()
train["Education"].value_counts()
train["Education"].value_counts(normalize=True) * 100

print("1(d) Find the count how many loan applicants are graduate and non graduate.")
print(" ")
train["Education"].value_counts(normalize=True).plot.bar(title="Education")
print(train["Education"].value_counts(normalize=True) * 100)
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()

train["Property_Area"].count()
train["Property_Area"].value_counts()
train["Property_Area"].value_counts(normalize=True) * 100

print("1(e) Find out the count how many loan applicants property lies in urban, rural and semi-urban areas.")
print(" ")
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

print("3: ")
print(
    "To visualize and plot the distribution plot of all numerical attributes of the given train dataset i.e. "
    "ApplicantIncome,  CoApplicantIncome and LoanAmount.     ")

print(" ")
print("ApplicantIncome distribution: ")
print(train["ApplicantIncome"])
plt.figure(1)
plt.subplot(121)
sns.distplot(train["ApplicantIncome"])

plt.subplot(122)
train["ApplicantIncome"].plot.box(figsize=(16, 5))
plt.show()

train.boxplot(column='ApplicantIncome', by="Education")
plt.suptitle(" ")
plt.show()

print("Coapplicant Income distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(train["CoapplicantIncome"])

plt.subplot(122)
train["CoapplicantIncome"].plot.box(figsize=(16, 5))
plt.show()

print("Loan Amount Variable")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df['LoanAmount'])

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16, 5))

plt.show()
print("Loan Amount Term Distribution")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df["Loan_Amount_Term"])

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16, 5))
plt.show()
