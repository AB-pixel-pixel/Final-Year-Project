import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'experiment': ['ε = 0.3', 'ε = 0.2', 'ε = 0.1'],
    'avg_finish': [0.5458333333333333, 0.5458333333333334, 0.4625000000000001],  # Sample values
    'average_total_conflict_count': [0.5833333333333334, 0.875, 0.75],  # Sample values
    'Load Balancing': [0.4419417382415922, 0.05892556509887875, 0.058925565098879064],  # Sample values
    'average_distance_moved': [1162.9749999999642, 1174.362499999968, 1173.533333333302]  # Sample values
}

# Convert data to arrays for plotting
experiments = data['experiment']
avg_finish = data['avg_finish']
average_total_conflict_count = data['average_total_conflict_count']
Load_Balancing = data['Load Balancing']
average_distance_moved = data['average_distance_moved']
# ε = 0.3-greedy
# Set plot style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Ablation Study Results', fontsize=16)

# Plot avg_finish
sns.lineplot(x=experiments, y=avg_finish, marker='o', ax=axes[0, 0])
axes[0, 0].set_title('Average Finish')
axes[0, 0].set_ylabel('Avg Finish')

# Plot average_total_conflict_count
sns.lineplot(x=experiments, y=average_total_conflict_count, marker='o', ax=axes[0, 1])
axes[0, 1].set_title('Average Total Conflict Count')
axes[0, 1].set_ylabel('Avg Total Conflict Count')

# Plot junheng
sns.lineplot(x=experiments, y=Load_Balancing, marker='o', ax=axes[1, 0])
axes[1, 0].set_title('Load_Balancing')
axes[1, 0].set_ylabel('Load_Balancing')

# Plot average_distance_moved
sns.lineplot(x=experiments, y=average_distance_moved, marker='o', ax=axes[1, 1])
axes[1, 1].set_title('Average Distance Moved')
axes[1, 1].set_ylabel('Avg Distance Moved')

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()