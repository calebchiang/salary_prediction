import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('salary_data.csv')

X = df['YearsExperience'].values
y = df['Salary'].values

plt.scatter(X, y, color='green', marker='o')
plt.title("Years of Experience vs Salary")
plt.ylabel("Years of Experience")
plt.show()
