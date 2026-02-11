import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

df['age'] = df['age'].fillna(df['age'].median())

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=df, x='sex', y='survived', ax=axes[0], 
    palette='viridis', order=sorted(df['sex'].unique())
)
axes[0].set_title('Survival Rate by Gender')
axes[0].set_ylabel('Survival Probability')

sns.barplot(
    data=df, x='pclass', y='survived', ax=axes[1], 
    palette='magma', order=sorted(df['pclass'].unique())
)
axes[1].set_title('Survival Rate by Passenger Class')
axes[1].set_ylabel('Survival Probability')

plt.tight_layout()
plt.savefig('survival_rates.png') 

summary = df.groupby(['sex', 'pclass'])['survived'].mean().reset_index().to_csv('survival_data_flat.csv', index=False)

summary.to_csv('survival_matrix.csv')

print("Survival Probability Matrix:")
print(summary)