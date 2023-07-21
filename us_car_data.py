
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data
df = pd.read_csv('/Users/jeffchuang/Desktop/us_car_data.csv')

# Strip extra spaces from 'Electric Vehicle Type'
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].str.strip()

# Preprocess the data for brands
df_brand = df[(df['Electric Vehicle Type'] == 'Battery Electric Vehicle') & (df['Model Year'].between(1997, 2024))]
df_brand['Sales'] = 1

# Group the data by 'City', 'Model Year', and 'Brand', and sum the 'Sales'
df_brand_grouped = df_brand.groupby(['City', 'Model Year', 'Make']).sum().reset_index()

# Use pivot_table to get the total sales for each brand in each city in each year
df_pivot_brand = df_brand_grouped.pivot_table(index='Model Year', columns='Make', values='Sales', aggfunc='sum').fillna(0)

# Get the data as a numpy array
data_brand = df_pivot_brand.to_numpy().T

# Preprocess the data for models
df_model = df[df['Model Year'].between(1997, 2024)]
df_model['Sales'] = 1

# Group the data by 'City', 'Model Year', and 'Model', and sum the 'Sales'
df_model_grouped = df_model.groupby(['City', 'Model Year', 'Model']).sum().reset_index()

# Use pivot_table to get the total sales for each model in each city in each year
df_pivot_model = df_model_grouped.pivot_table(index='Model Year', columns='Model', values='Sales', aggfunc='sum').fillna(0)

# Get the data as a numpy array
data_model = df_pivot_model.to_numpy().T

# Create a figure
fig = plt.figure(figsize=(20, 10))

# Create the first subplot
# [left, bottom, width, height] in the figure coordinate, which ranges from 0 to 1
ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])

# Create the second subplot in the lower right corner
# [left, bottom, width, height] in the figure coordinate, which ranges from 0 to 1
ax2 = fig.add_axes([0.65, 0.1, 0.3, 0.4])

years_brand = df_pivot_brand.index.unique()  # Get the unique years for brands
years_model = df_pivot_model.index.unique()  # Get the unique years for models

num_frames_brand = data_brand.shape[1]  # Set the number of frames for the brand animation
num_inter_frames_brand = 10  # Insert more frames between each brand animation frame to create a sliding effect

num_frames_model = data_model.shape[1]  # Set the number of frames for the model animation
num_inter_frames_model = 10  # Insert more frames between each model animation frame to create a sliding effect

# Define the animation update function for brands
def update_brand(num):
    ax1.clear()
    year, inter_frame = divmod(num, num_inter_frames_brand)  # Get the year and the number of inter frames
    if year < num_frames_brand:
        prev_data = data_brand[:, :year].sum(axis=1) if year > 0 else np.zeros(data_brand.shape[0])
        inter_data = data_brand[:, year] / num_inter_frames_brand * inter_frame  # Calculate the inter data
        cumulative_data = prev_data + inter_data  # Calculate the cumulative data
        rank_order = np.argsort(cumulative_data)  # Get the sorting indices

        # Generate bar colors using different color maps
        cmap = plt.get_cmap('tab10')
        colors = cmap(rank_order % 10)

        bars = ax1.barh(df_pivot_brand.columns[rank_order], cumulative_data[rank_order], height=0.5, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlim([0, max(cumulative_data) + 50])  # Adjust the x-axis range based on the maximum cumulative data

        # Add the numbers to the right side of the bars
        for bar, value in zip(bars, cumulative_data[rank_order]):
            if value != 0:
                ax1.text(value + 0.3, bar.get_y() + bar.get_height() / 2, str(int(value)), va='center', ha='left', color='black')

        ax1.set_title(f"Cumulative Electric Car Brand production up to {years_brand[year]}")

# Define the animation update function for models
def update_model(num):
    ax2.clear()
    year, inter_frame = divmod(num, num_inter_frames_model)  # Get the year and the number of inter frames
    if year < num_frames_model:
        prev_data = data_model[:, :year].sum(axis=1) if year > 0 else np.zeros(data_model.shape[0])
        inter_data = data_model[:, year] / num_inter_frames_model * inter_frame  # Calculate the inter data
        cumulative_data = prev_data + inter_data  # Calculate the cumulative data
        rank_order = np.argsort(cumulative_data)  # Get the sorting indices

        # Select the top 30 models
        rank_order = rank_order[-30:]

        # Generate bar colors using different color maps
        cmap = plt.get_cmap('tab10')
        colors = cmap(rank_order % 10)

        bars = ax2.barh(df_pivot_model.columns[rank_order], cumulative_data[rank_order], height=0.5, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlim([0, max(cumulative_data) + 50])  # Adjust the x-axis range based on the maximum cumulative data

        # Add the numbers to the right side of the bars
        for bar, value in zip(bars, cumulative_data[rank_order]):
            if value != 0:
                ax2.text(value + 0.3, bar.get_y() + bar.get_height() / 2, str(int(value)), va='center', ha='left', color='black')

        ax2.set_title(f"Cumulative Electric Car Model production up to {years_model[year]}")

# Create the first animation
ani_brand = animation.FuncAnimation(fig, update_brand, frames=range((num_frames_brand - 1) * num_inter_frames_brand + 1), interval=100, repeat=False)

# Create the second animation
ani_model = animation.FuncAnimation(fig, update_model, frames=range((num_frames_model - 1) * num_inter_frames_model + 1), interval=100, repeat=False)

plt.tight_layout()
plt.show()


