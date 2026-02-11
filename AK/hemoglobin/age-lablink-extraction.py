# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the CSV file
# file_path = "/home/apc-3/PycharmProjects/PythonProjectAK/lablink-nov2024-1k-hb/lablink_1k_hemoglobin_age_gender.csv"
# df = pd.read_csv(file_path)
#
# # Clean the 'Age' column: remove 'Y' and convert to integers
# df['Age'] = df['Age'].str.replace('Y', '').astype(int)
#
# # Define age group bins and labels
# bins = [19, 29, 39, 49, 59, 69, 79]
# labels = ['20s', '30s', '40s', '50s', '60s', '70s']
#
# # Create a new column 'AgeGroup'
# df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
#
# # Count the number of samples per age group
# age_group_counts = df['AgeGroup'].value_counts().sort_index()
#
# # Create the bar plot
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(x=age_group_counts.index, y=age_group_counts.values, color='skyblue')
# plt.title('Age Group Distribution')
# plt.xlabel('Age Group')
# plt.ylabel('Count')
#
# # Annotate each bar with its count
# for i, count in enumerate(age_group_counts.values):
#     ax.annotate(f'{count}', (i, count), ha='center', va='bottom', fontsize=10)
#
# plt.tight_layout()
#
# # Save the plot as an image
# plt.savefig("age_group_distribution_bar_chart.png")
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Load the CSV file
file_path = "/home/apc-3/PycharmProjects/PythonProjectAK/lablink-nov2024-1k-hb/lablink_1k_hemoglobin_age_gender.csv"
df = pd.read_csv(file_path)

# Clean the 'Age' column: remove 'Y' and convert to integers
df['Age'] = df['Age'].str.replace('Y', '').astype(int)

# Define age group bins and labels
bins = [19, 29, 39, 49, 59, 69, 79]
labels = ['20s', '30s', '40s', '50s', '60s', '70s']

# Create a new column 'AgeGroup'
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Count the number of samples per age group
age_group_counts = df['AgeGroup'].value_counts().sort_index()

# Create the bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=age_group_counts.index, y=age_group_counts.values, color='skyblue')
plt.title('Age Group Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')

# Annotate each bar with its count
for i, count in enumerate(age_group_counts.values):
    ax.annotate(f'{count}', (i, count), ha='center', va='bottom', fontsize=10)

# Define a custom legend to explain age group ranges
legend_labels = [
    Patch(color='skyblue', label='20s: 20–29'),
    Patch(color='skyblue', label='30s: 30–39'),
    Patch(color='skyblue', label='40s: 40–49'),
    Patch(color='skyblue', label='50s: 50–59'),
    Patch(color='skyblue', label='60s: 60–69'),
    Patch(color='skyblue', label='70s: 70–79'),
]

# Add the legend to the plot
plt.legend(handles=legend_labels, title="Age Group Ranges", loc='upper right')

plt.tight_layout()

# Save the plot as an image
plt.savefig("age_group_distribution_bar_chart_with_legend.png")
plt.show()
