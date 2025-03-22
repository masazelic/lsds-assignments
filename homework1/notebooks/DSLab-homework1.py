# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSLab Homework 1 - Data Science with CO2
#
# ## Hand-in Instructions
#
# - __Due: 19.03.2024 23h59 CET__
# - `./setup.sh` before you can start working on this notebook.
# - `git push` your final verion to the master branch of your group's Renku repository before the due date.
# - check if `environment.yml` and `requirements.txt` are properly written
# - add necessary comments and discussion to make your codes readable

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Carbosense
#
# The project Carbosense establishes a uniquely dense CO2 sensor network across Switzerland to provide near-real time information on man-made emissions and CO2 uptake by the biosphere. The main goal of the project is to improve the understanding of the small-scale CO2 fluxes in Switzerland and concurrently to contribute to a better top-down quantification of the Swiss CO2 emissions. The Carbosense network has a spatial focus on the City of Zurich where more than 50 sensors are deployed. Network operations started in July 2017.
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#
# <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_ZLMT_3.JPG" width="156">  <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_sensor_SMALL.jpg" width="300">

# %% [markdown]
# ## Description of the homework
#
# In this homework, we will curate a set of **CO2 measurements**, measured from cheap but inaccurate sensors, that have been deployed in the city of Zurich from the Carbosense project. The goal of the exercise is twofold:
#
# 1. Learn how to deal with real world sensor timeseries data, and organize them efficiently using python dataframes.
#
# 2. Apply data science tools to model the measurements, and use the learned model to process them (e.g., detect drifts in the sensor measurements).
#
# The sensor network consists of 46 sites, located in different parts of the city. Each site contains three different sensors measuring (a) **CO2 concentration**, (b) **temperature**, and (c) **humidity**. Beside these measurements, we have the following additional information that can be used to process the measurements:
#
# 1. The **altitude** at which the CO2 sensor is located, and the GPS coordinates (latitude, longitude).
#
# 2. A clustering of the city of Zurich in 17 different city **zones** and the zone in which the sensor belongs to. Some characteristic zones are industrial area, residential area, forest, glacier, lake, etc.
#
# ## Prior knowledge
#
# The average value of the CO2 in a city is approximately 400 ppm. However, the exact measurement in each site depends on parameters such as the temperature, the humidity, the altitude, and the level of traffic around the site. For example, sensors positioned in high altitude (mountains, forests), are expected to have a much lower and uniform level of CO2 than sensors that are positioned in a business area with much higher traffic activity. Moreover, we know that there is a strong dependence of the CO2 measurements, on temperature and humidity.
#
# Given this knowledge, you are asked to define an algorithm that curates the data, by detecting and removing potential drifts. **The algorithm should be based on the fact that sensors in similar conditions are expected to have similar measurements.**
#
# ## To start with
#
# The following csv files in the `../data/carbosense-raw/` folder will be needed:
#
# 1. `CO2_sensor_measurements.csv`
#
#    __Description__: It contains the CO2 measurements `CO2`, the name of the site `LocationName`, a unique sensor identifier `SensorUnit_ID`, and the time instance in which the measurement was taken `timestamp`.
#
# 2. `temperature_humidity.csv`
#
#    __Description__: It contains the temperature and the humidity measurements for each sensor identifier, at each timestamp `Timestamp`. For each `SensorUnit_ID`, the temperature and the humidity can be found in the corresponding columns of the dataframe `{SensorUnit_ID}.temperature`, `{SensorUnit_ID}.humidity`.
#
# 3. `sensor_metadata_updated.csv`
#
#    __Description__: It contains the name of the site `LocationName`, the zone index `zone`, the altitude in meters `altitude`, the longitude `LON`, and the latitude `LAT`.
#
# Import the following python packages:

# %%
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go
import os

# %%
pd.options.mode.chained_assignment = None

# %% [markdown]
# ## PART I: Handling time series with pandas (10 points)

# %%
# Set the data directory
DATA_DIR = "../data/"

# %%
# Load the data
co2_sensor = pd.read_csv(DATA_DIR + "CO2_sensor_measurements.csv", sep="\t")
sensors_metadata = pd.read_csv(DATA_DIR + "sensors_metadata_updated.csv", index_col=0)
temperature_humidity = pd.read_csv(DATA_DIR + "temperature_humidity.csv", sep="\t")

# %%
display(co2_sensor.head())


# %%
display(sensors_metadata.head())


# %%
display(temperature_humidity.head())


# %% [markdown]
# ### a) **8/10**
#
# Merge the `CO2_sensor_measurements.csv`, `temperature_humidity.csv`, and `sensors_metadata.csv`, into a single dataframe.
#
# * The merged dataframe contains:
#     - index: the time instance `timestamp` of the measurements
#     - columns: the location of the site `LocationName`, the sensor ID `SensorUnit_ID`, the CO2 measurement `CO2`, the `temperature`, the `humidity`, the `zone`, the `altitude`, the longitude `lon` and the latitude `lat`.
#
# | timestamp | LocationName | SensorUnit_ID | CO2 | temperature | humidity | zone | altitude | lon | lat |
# |:---------:|:------------:|:-------------:|:---:|:-----------:|:--------:|:----:|:--------:|:---:|:---:|
# |    ...    |      ...     |      ...      | ... |     ...     |    ...   |  ... |    ...   | ... | ... |
#
#
#
# * For each measurement (CO2, humidity, temperature), __take the average over an interval of 30 min__.
#
# * If there are missing measurements, __interpolate them linearly__ from measurements that are close by in time.
#
# __Hints__: The following methods could be useful
#
# 1. ```python
# pandas.DataFrame.resample()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
#
# 2. ```python
# pandas.DataFrame.interpolate()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
#
# 3. ```python
# pandas.DataFrame.mean()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#
# 4. ```python
# pandas.DataFrame.append()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

# %% [markdown]
# ### Preprocessing 

# %%
# Convert the "timestamp" column to datetime format
co2_sensor["timestamp"] = pd.to_datetime(co2_sensor["timestamp"])
temperature_humidity["Timestamp"] = pd.to_datetime(temperature_humidity["Timestamp"])

# Set the timestamp as the index
co2_sensor = co2_sensor.set_index("timestamp")

# Take the average over 30min for CO2
co2_sensor_resampled = (
    co2_sensor.groupby(["SensorUnit_ID", "LocationName"]).resample("30T").mean()
)

# Reset the index
co2_sensor_resampled = co2_sensor_resampled.reset_index()

# Vizualize the resampled data
display(co2_sensor_resampled.head())


# %%
# Reshape the temperature and humidity dataframe for merging
temperature_humidity = temperature_humidity.melt(
    id_vars=["Timestamp"], var_name="SensorUnit_ID", value_name="Measurement"
)

# Split the SensorUnit_ID and measurement type into seperate columns
temperature_humidity[["SensorUnit_ID", "Measurement_Type"]] = temperature_humidity[
    "SensorUnit_ID"
].str.split(".", expand=True)

# Pivot the table to have separate columns for temperature and humidity
temperature_humidity = temperature_humidity.pivot(
    index=["Timestamp", "SensorUnit_ID"],
    columns="Measurement_Type",
    values="Measurement",
).reset_index()

# Convert SensorUnit_ID to integer values
temperature_humidity["SensorUnit_ID"] = temperature_humidity["SensorUnit_ID"].astype(int)

# Set Timestamp as index
temperature_humidity.set_index("Timestamp", inplace=True)

# Group by SensorUnit_ID and resample to calculate the mean for each 30-minute interval like in co2 table
temperature_humidity_resampled = (
    temperature_humidity.groupby("SensorUnit_ID").resample("30T").mean()
)
temperature_humidity_resampled = temperature_humidity_resampled.drop(
    "SensorUnit_ID", axis=1
)
temperature_humidity_resampled = temperature_humidity_resampled.reset_index()

display(temperature_humidity_resampled.head())

# %% [markdown]
# ### Merging Dataframes

# %%
# Merge co2_sensor_resampled with temperature_humidity_resampled based one timestamp and SensorUnit_ID
merged_1 = co2_sensor_resampled.merge(
    temperature_humidity_resampled,
    how="left",
    left_on=["timestamp", "SensorUnit_ID"],
    right_on=["Timestamp", "SensorUnit_ID"],
)
merged_1.drop(columns="Timestamp", inplace=True)

display(merged_1.head())

# %%
# Then merge with sensors_metadata
sensors_metadata.drop(columns=["X", "Y"], inplace=True)
sensors_metadata.rename(columns={"LAT": "lat", "LON": "lon"}, inplace=True)
merged_2 = merged_1.merge(sensors_metadata, how="left", on="LocationName")

display(merged_2.head())

# %%
# Interpolate the missing values
merged_data = merged_2.interpolate(method="linear")

display(merged_data.head())

# %% [markdown]
# ### b) **2/10**
#
# Export the curated and ready to use timeseries to a csv file, and properly push the merged csv to Git LFS.

# %%

# %% [markdown]
# ## PART II: Data visualization (15 points)

# %% [markdown]
# ### a) **5/15**
# Group the sites based on their altitude, by performing K-means clustering.
# - Find the optimal number of clusters using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)).
# - Wite out the formula of metric you use for Elbow curve.
# - Perform clustering with the optimal number of clusters and add an additional column `altitude_cluster` to the dataframe of the previous question indicating the altitude cluster index.
# - Report your findings.
#
# __Note__: [Yellowbrick](http://www.scikit-yb.org/) is a very nice Machine Learning Visualization extension to scikit-learn, which might be useful to you.

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from yellowbrick.cluster import KElbowVisualizer

# %% [markdown]
# ### Find the optimal number of cluster by using the Elbow method is: {optimal_k}")

# %%
# Standardize the altitude column
scaler = StandardScaler().fit(merged_data[["altitude"]])
merged_data["altitude_scaled"] = scaler.transform(merged_data[["altitude"]])

# Initialize the KMeans model 
model = KMeans(n_init=10, max_iter=300, random_state=42)

# Use the elbow method to find the optimal number of clusters
visualizer = KElbowVisualizer(
    model, k=(1,11), metric = 'distortion', timings=False, locate_elbow=True
)

# Fit the data to the visualizer
visualizer.fit(merged_data[["altitude_scaled"]])      
visualizer.show() 
optimal_k = visualizer.elbow_value_

# Print the optimal number of clusters
print(f"The optimal number of clusters is: {optimal_k}")

# %% [markdown]
# ### Formula of metric of our Elbow curve.
#
# The metric that we used for the elbow curve method of obtaining the optimal number of clusters is **Mean Distortion**, which basically corresponds to the sum of squared distances between each observation and its closest centroid.
# $$mean\_distortion = \sum_{i=1}^k \sum_{x \in C_i} ||x-\mu_i||^2$$
#
# where k is the number of clusters, $C_i$ represents the set of all points (or data points) assigned to cluster $i$, x is a point in cluster $C_i$ and $\mu_i$ is the centroid of the cluster $C_i$.

# %%
# Initialize the KMeans clustering algorithm with the optimal number of clusters.
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)

# Fit the model to the dat and predict the cluster index for each sample, then add this as a new column 'altitude_cluster' in the dataframe.
merged_data["altitude_cluster"] = kmeans.fit_predict(merged_data[["altitude_scaled"]])

# Reassign cluster indices based on altitude to make them more intuitive: Clusters are sorted by their mean altitude (from lowest to highest). 
# This means cluster 0 will have the lowest mean altitude, and the highest cluster number will have the highest mean altitude.
cluster_means = merged_data.groupby('altitude_cluster')['altitude_scaled'].mean()
sorted_cluster_indexes = cluster_means.sort_values().index
cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_indexes)}
merged_data["altitude_cluster"] = merged_data["altitude_cluster"].map(cluster_mapping)

# Convert the 'altitude_cluster' column to a categorical type for easier analysis and visualization.
merged_data['altitude_cluster_cat'] = merged_data["altitude_cluster"].astype("category")

display(merged_data.head())

# %% [markdown]
# ### Report of our findings

# %%
# Extract and clean up unique site data, including location, altitude, cluster, and coordinates, for clear visualization.
cluster_site = merged_data[['LocationName', 'altitude', 'altitude_cluster_cat', 'lon', 'lat']].drop_duplicates()
cluster_site.reset_index(drop=True, inplace=True)
display(cluster_site)

# %%
# Groups the data by altitude cluster category and calculates the specified metrics for each group.
cluster_info = cluster_site.groupby('altitude_cluster_cat').agg(
    Number_of_sites=('LocationName', 'count'),
    altitude_min=('altitude', 'min'),
    altitude_max=('altitude', 'max'),
    altitude_mean=('altitude', 'mean'), 
    altitude_std=('altitude', 'std')
).reset_index()

display(cluster_info)

# %% [markdown]
# Observing the results from the table, we can say that:
#
# **Cluster 0**: With the highest number of sites (36), this cluster seems to represent areas at relatively lower altitudes, ranging from 391.9 to 490.2, with a mean altitude of 427.25. The standard deviation of approximately 24.05 suggests that the sites within this cluster have altitudes that are relatively similar, indicating less variability in the altitude of these locations.
#
# **Cluster 1**: This cluster has fewer sites (9) compared to Cluster 0, and it occupies a higher altitude band ranging from 510.5 to 683.9, with a mean altitude of 582.49. The larger standard deviation of 62.23 compared to Cluster 0 implies that there is a greater spread or variability in the altitude of the sites within this cluster.
#
# **Cluster 2**: The single site in this cluster is at a significantly higher altitude than the other clusters (863.6). Given that there's only one site, the standard deviation is not applicable. 
#
# Based on these observations, the clustering seems to have effectively grouped the sites into distinct tiers based on altitude. Cluster 0 could be considered as low altitude, Cluster 1 as mid altitude, and Cluster 2 as high altitude.
#
# The scatter plot below allows us to better observe the clusters based on altitude, as well as the distribution of these clusters.

# %%
df_sorted = cluster_site.sort_values('altitude')

# Create a bar chart using Plotly to visualize the altitude of sites. 
fig = px.bar(df_sorted, x='LocationName', y='altitude',
             color='altitude_cluster_cat', 
             title='Altitude of Sites by Location Name and Cluster Category',
             labels={'altitude': 'Altitude (meters)', 'LocationName': 'Site Name'})

# Update the layout of the figure to make it more readable.
fig.update_layout(xaxis={'categoryorder':'total ascending'}, xaxis_title="Site Name", yaxis_title="Altitude (meters)")
fig.update_layout(legend_title_text='Altitude Cluster Category')
fig.update_xaxes(tickangle=45)

fig.show()

# %% [markdown]
# The scatter plot below shows us the distribution of clusters based on their geographic positions in Zurich.
#
# Cluster 0, which contains the largest number of sites, is distributed quite homogeneously both in terms of altitude—as demonstrated by the graph above—and geographically, indicating a common low-urban area.
#
# Cluster 1, with fewer sites, exhibits a wider variation in altitudes, which could represent somewhat higher areas of the city.
#
# As for the single site in Cluster 2, its significantly higher altitude might place it in a special category, such as a significant elevated area relative to the general urban spread, like a viewpoint or a natural attraction.
#
# These clusters therefore illustrate a possible vertical division of the urban areas of Zurich.

# %%
# Scatter plot with Plotly 
fig = px.scatter(cluster_site, x='lon', y='lat',
                 color='altitude_cluster_cat', 
                 size='altitude', 
                 hover_name='LocationName', 
                 hover_data=['altitude'], 
                 title='Altitude Clusters Distribution per Site in 2D')

# Update the layout of the figure 
fig.update_layout(legend_title_text='Altitude Cluster')
fig.update_xaxes(title_text='Longitude')
fig.update_yaxes(title_text='Latitude')

fig.show()

# %% [markdown]
# ### b) **4/15**
#
# Use `plotly` (or other similar graphing libraries) to create an interactive plot of the monthly median CO2 measurement for each site with respect to the altitude.
#
# Add proper title and necessary hover information to each point, and give the same color to stations that belong to the same altitude cluster.

# %%
# Calculate the median CO2 levels for each site within each altitude cluster.
CO2_median = (
    merged_data.groupby(["LocationName", "altitude", "altitude_cluster_cat"])
    .agg({"CO2": "median"})
    .reset_index()
)

fig = px.scatter(
    CO2_median,
    x="CO2",
    y="altitude",
    color="altitude_cluster_cat",
    labels={'CO2': 'CO2 [ppm]' , 'altitude': 'Altitude [m]', 'LocationName': 'Site', 'altitude_cluster': 'Altitude Cluster'},
    hover_data=["LocationName"],
)

fig.update_layout(
    title=go.layout.Title(
        text="Monthly median CO2 measurement for each site w.r.t the altitude"
    ), 
    legend_title_text='Altitude Cluster'
)

# %% [markdown]
# ### c) **6/15**
#
# Use `plotly` (or other similar graphing libraries) to plot an interactive time-varying density heatmap of the mean daily CO2 concentration for all the stations. Add proper title and necessary hover information.
#
# __Hints:__ Check following pages for more instructions:
# - [Animations](https://plotly.com/python/animations/)
# - [Density Heatmaps](https://plotly.com/python/mapbox-density-heatmaps/)

# %%
# Add new 'day' column to the table
merged_data.timestamp = pd.to_datetime(merged_data.timestamp)
merged_data["day"] = merged_data.timestamp.dt.day

# Group by day, location, altitude, altitude_cluster, latitude, and longitude, and calculate mean CO2 concentration
CO2_mean_daily = (
    merged_data.groupby(
        ["day", "LocationName", "altitude", "altitude_cluster", "lat", "lon"]
    )
    .agg({"CO2": "mean"})
    .reset_index()
)

# Calculate center point for map
center = dict(lat=CO2_mean_daily["lat"].mean(), lon=CO2_mean_daily["lon"].mean())

# Create density map animation
fig2 = px.density_mapbox(
    CO2_mean_daily,
    lat="lat",
    lon="lon",
    z="CO2",
    animation_frame="day",
    hover_data=["LocationName"],
    radius=20,
    center=center,
    zoom=10,
    mapbox_style="open-street-map",
)

# Update layout of the figure
fig2.update_layout(
    title=go.layout.Title(text="Mean daily CO2 concentration for all the stations")
)

fig2.show()


# %% [markdown]
# ## PART III: Model fitting for data curation (35 points)

# %% [markdown]
# ### a) **2/35**
#
# The domain experts in charge of these sensors report that one of the CO2 sensors `ZSBN` is exhibiting a drift on Oct. 24. Verify the drift by visualizing the CO2 concentration of the drifting sensor and compare it with some other sensors from the network.

# %% [markdown]
# ### Vizualization of the CO2 concentration of the drifting sensor

# %%
# Data Preprocessing : Selection of the drifting sensors and of the features of interest

drifting_sensor = 'ZSBN'
sensor_ZSBN = merged_data[merged_data["LocationName"] == drifting_sensor][
    ["timestamp", "CO2", "humidity", "temperature"]
].reset_index(inplace=False)

display(sensor_ZSBN.head())

# %%
# Define the start date of sensor drift
drifting_start = "2017-10-24 00:00:00"

# Create a line plot for CO2 concentration over time for sensor ZSBN
fig = px.line(sensor_ZSBN, x='timestamp', y='CO2', title='Evolution of CO2 concentration over time for sensor ZSBN',
              labels={'timestamp': 'Time', 'CO2': 'CO2 Concentration [ppm]'}, line_shape='linear')

# Add a red dashed vertical line to indicate the start of drift
fig.add_vline(x=drifting_start, line_width=3, line_dash="dash", line_color="red")

# Annotate the drifting start date on the plot
fig.add_annotation(x=drifting_start, y=sensor_ZSBN[sensor_ZSBN['timestamp'] == drifting_start]['CO2'].values[0],
                   text="Drifting day", showarrow=True, arrowhead=1, ax=-50, ay=-100)

# Set x-axis ticks to be daily
fig.update_xaxes(dtick=24 * 60 * 60 * 1000)

# Display the figure
fig.show()

# %% [markdown]
# The figure below displays the evolution of the CO2 concentration measured by the sensor identified as having drifted. Notably, there is a substantial decrease in the recorded CO2 levels beginning on October 24th, which clearly illustrates the occurrence of drift in the sensor's readings.

# %% [markdown]
# #### **Comparison with other sensors of the network**
#
# To compare the drifting sensor with others in the network, I first standardized the CO2 values for each site to make them comparable. I then grouped the data by `drifting_status` and `timestamp`, calculating the mean of the standardized CO2 concentrations. This approach enabled me to visualize the average behavior of drifting versus non-drifting sensors over time, providing a clear comparison of their performance and highlighting any significant differences in their recorded CO2 levels.

# %%
from sklearn.preprocessing import StandardScaler

df_comparison = merged_data[["timestamp", "LocationName", "CO2", "humidity", "temperature"]]

# Assign drifting status based on LocationName
df_comparison['drifting_status'] = ['Drifting sensor' if sensor == drifting_sensor else 'Non drifting sensor' for sensor in df_comparison['LocationName']]

# Standardize the CO2 measurements
scaler = StandardScaler()
def standardize_data(group):
    group['CO2_standardized'] = scaler.fit_transform(group[['CO2']])
    return group

# Apply standardization function to each group
df_comparison = df_comparison.groupby('LocationName').apply(standardize_data).reset_index(drop=True)

# Group by drifting status and timestamp and calculate the mean of CO2 measurements
df_comparison = df_comparison.groupby(['drifting_status', 'timestamp']).agg({'CO2_standardized': 'mean'}).reset_index()

display(df_comparison.head())

# %%
# Create a line plot for CO2 concentration over time depending on the sensor status
fig = px.line(df_comparison, x='timestamp', y='CO2_standardized', color='drifting_status',
              title='Evolution of CO2 concentration over time by sensor status',
              labels={'timestamp': 'Time', 'CO2_standardized': 'Normalized CO2 Concentration', 'drifting_status': 'Sensor Status'})

# Add a vertical line for the start of drifting and an annotation for the drifting day
fig.add_vline(x=drifting_start, line_width=3, line_dash="dash", line_color="red")
fig.add_annotation(x=drifting_start, y=0,
                   text="Drifting day", showarrow=True, arrowhead=1, ax=-50, ay=-100)

# Set x-axis ticks to be daily
fig.update_xaxes(dtick=24 * 60 * 60 * 1000)

# Display the figure
fig.show()

# %% [markdown]
# The graph above illustrates the temporal evolution of the standardized mean CO2 concentration from non-drifting sites (in red) alongside the drifting site (in blue). Before the drifting day, the sensor that would later experience drift typically registers above the average of other sensors. However, from the drifting day forward, we see a reversal in this trend— the standardized CO2 value of the drifting sensor drops below the average of the others, unequivocally indicating the drift.

# %% [markdown]
# ### b) **8/35**
#
# The domain experts ask you if you could reconstruct the CO2 concentration of the drifting sensor had the drift not happened. You decide to:
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features the covariates not affected by the malfunction (such as temperature and humidity)
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __95% confidence interval__ obtained from cross validation: assume that the error follows a normal distribution and is independent of time.
# - What do you observe? Report your findings.
#
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %% [markdown]
# ### Fit a linear regression model to the CO2 measurements of the drifting site considering `humdidity` and `temperature` as features

# %%
from sklearn.linear_model import LinearRegression

# Data Preprocessing : Selection of the drifting sensors and of the features of interest
sensor_ZSBN = merged_data[merged_data["LocationName"] == "ZSBN"][
    ["timestamp", "CO2", "humidity", "temperature"]
]

display(sensor_ZSBN.head())

# %%
# Selection of the data before the drifting date in order to train our linear regressor
drifting_day = 24

# Split the data into features and target
X = sensor_ZSBN[sensor_ZSBN.timestamp.dt.day < drifting_day][
    ["humidity", "temperature"]
]
y = sensor_ZSBN[sensor_ZSBN.timestamp.dt.day < drifting_day][["CO2"]]

# Fitting the model
model = LinearRegression().fit(X, y)

# %% [markdown]
# ### Interactive CO2 Plot with Predictions and Confidence Intervals

# %%
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy.stats import norm

# Cross-validation using TimeSeriesSplit
tscv = TimeSeriesSplit()
scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")

# Calculate root mean square errors
rmse_errors = np.sqrt(-scores)

# Standard deviation of the errors
std_error = np.std(rmse_errors)

# Predict the CO2 levels of the full dataset
sensor_ZSBN["CO2_predicted"] = np.round(
    model.predict(sensor_ZSBN[["humidity", "temperature"]]), decimals=4
)

# Calculate the 95% confidence interval
confidence_interval = norm.interval(
    0.95, loc=sensor_ZSBN["CO2_predicted"], scale=std_error
)
sensor_ZSBN["confidence_interval_lower"] = np.round(confidence_interval[0], decimals=4)
sensor_ZSBN["confidence_interval_upper"] = np.round(confidence_interval[1], decimals=4)

# %%
# Create the interactive plot with actual CO2 measurements, predicted values and the 95% confidence interval
fig = go.Figure()

# Add trace for actual CO2 measurements
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN["timestamp"],
        y=sensor_ZSBN["CO2"],
        mode="lines",
        name="Actual CO2",
        hoverinfo="text",
        text=("Actual CO2: " + np.round(sensor_ZSBN["CO2"], decimals=4).astype(str)),
    )
)

# Add trace for predicted CO2 values
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN["timestamp"],
        y=sensor_ZSBN["CO2_predicted"],
        mode="lines",
        name="Predicted CO2",
        hoverinfo="text",
        text=[
            f"Date: {date.strftime('%b %d, %Y %I:%M %p')}<br><br>Predicted CO2: {pred}<br>Upper CI: {upper}<br>Lower CI: {lower}"
            for date, upper, pred, lower in zip(
                pd.to_datetime(sensor_ZSBN["timestamp"]),
                sensor_ZSBN["CO2_predicted"],
                sensor_ZSBN["confidence_interval_upper"],
                sensor_ZSBN["confidence_interval_lower"],
            )
        ],
    )
)

# Add trace for the lower bound of the confidence interval
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN["timestamp"],
        y=confidence_interval[0],
        line=dict(width=0),
        hoverinfo="none",
        showlegend=False,
    )
)

# Add trace for the upper bound of the confidence interval
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN["timestamp"],
        y=confidence_interval[1],
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        hoverinfo="none",
        showlegend=False,
    )
)
# Add a vertical line for the start of drifting and an annotation for the drifting day
fig.add_vline(x=drifting_start, line_width=3, line_dash="dash", line_color="red")
fig.add_annotation(x=drifting_start, y=sensor_ZSBN[sensor_ZSBN['timestamp'] == drifting_start]['CO2'].values[0],
                   text="Drifting day", showarrow=True, arrowhead=1, ax=-50, ay=-100)

# Update the layout of the figure
fig.update_layout(
    title="CO2 Level Prediction with 95% Confidence Interval",
    xaxis_title="Date",
    yaxis_title="CO2 Level (ppm)",
    hovermode="x unified",
    showlegend=True,
)

fig.update_xaxes(dtick=24 * 60 * 60 * 1000)

fig.show()

# %% [markdown]
# ### Observations and Findings: Analysis Report

# %% [markdown]
# The chart presented illustrates the temporal evolution of the CO2 level from the ZSBN sensor along with the values predicted by our linear regression model. We notice a daily periodicity in the CO2 variations captured by the sensor, which our model predicts with a certain degree of accuracy. Notably, we observe daily minimums around noon and maximums around 7 a.m.
#
# The "drift" on October 24th is clearly visible here. By examining the blue curve, representing the actual values, we perceive a significant drop in CO2 level on that day, which persists until the end of the month. Despite this sharp decrease, the periodicity of the measurements remains evident with morning peaks and a dip around noon
#
#
# Our model seems to capture this daily oscillation of CO2 well. However, it tends to smooth the results. Indeed, we sometimes observe significant spikes of CO2 in the values recorded by the sensor, which could correspond, for example, to morning traffic jams. The model manages to capture the CO2 variation during a typical day but cannot obviously predict these spikes with the features we have provided for trainig.
#
# After the drift of October 24th, the predictions remain consistent, preserving the observed periodicity and staying within a plausible range of values in light of previous measurements at the site. This consistency suggests that the model, despite its limitations, offers a reliable reflection of the CO2 concentration trends.

# %% [markdown]
# ### c) **10/35**
#
# In your next attempt to solve the problem, you decide to exploit the fact that the CO2 concentrations, as measured by the sensors __experiencing similar conditions__, are expected to be similar.
#
# - Find the sensors sharing similar conditions with `ZSBN`. Explain your definition of "similar condition".
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features:
#     - the information of provided by similar sensors
#     - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.

# %% [markdown]
# ### Exploring Similar Conditions: Identifying CO2 Sensor Relationships with ZSBN

# %% [markdown]
# We use the CO2 measurements from sensors having a similar altitude (i.e. being in the same altitude cluster), and located in the same type of zone.
# This is motivated by the fact that sensors sharing these same attributes are usually exposed to the same conditions affecting the CO2 concentration, as indicated in the introduction.

# %%
# We first identify the altitude cluster and the zone of the ZSBN sensor
altitude_cluster, zone = (
    merged_data[["altitude_cluster", "zone"]]
    .loc[merged_data["LocationName"] == drifting_sensor]
    .values[0]
)

# we select the sensors with similar conditions
similar_sensors = merged_data[
    (merged_data["altitude_cluster"] == altitude_cluster)
    & (merged_data["zone"] == zone)
    & (merged_data["LocationName"] != drifting_sensor)
].LocationName.unique()

print(f"There are {len(similar_sensors)} sensors with similar conditions:")
print(similar_sensors)

# %%
measurements_similar_sensors = merged_data[
    merged_data["LocationName"].isin(similar_sensors)
]

measurements_similar_sensors_pivot = measurements_similar_sensors.pivot(
    index="timestamp",
    columns="LocationName",
    values=["CO2"],
)

measurements_similar_sensors_pivot.columns = [
    f"{col[1]}_{col[0]}" for col in measurements_similar_sensors_pivot.columns
]

# we create a new df by joining the measurements of the similar sensors with the measurements of the ZSBN sensor
sensor_ZSBN_with_additional_data = measurements_similar_sensors_pivot.join(
    sensor_ZSBN[["humidity", "temperature", "timestamp", "CO2"]].set_index("timestamp")
).reset_index()

display(sensor_ZSBN_with_additional_data.head())

# %% [markdown]
# ### Linear Regression Model Incorporating Similar Sensor Data

# %%
# we fit a new model using the humidity and temperature measurements (as before) + the CO2 measurements of the similar sensors
X = (
    sensor_ZSBN_with_additional_data[
        sensor_ZSBN_with_additional_data.timestamp.dt.day < drifting_day
    ]
).drop(columns=["timestamp", "CO2"])

y = sensor_ZSBN_with_additional_data[
    sensor_ZSBN_with_additional_data.timestamp.dt.day < drifting_day
][["CO2"]]

# Fitting the model
model = LinearRegression().fit(X, y)

# %% [markdown]
# ### Interactive CO2 Plot with Predictions and Confidence Intervals

# %%
# Cross-validation using TimeSeriesSplit
tscv = TimeSeriesSplit()
scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")

# Calculate root mean square errors
rmse_errors = np.sqrt(-scores)

# Standard deviation of the errors
std_error = np.std(rmse_errors)

# Predict the CO2 levels of the full dataset
sensor_ZSBN_with_additional_data["CO2_predicted"] = np.round(
    model.predict(sensor_ZSBN_with_additional_data.drop(columns=["timestamp", "CO2"])),
    decimals=4,
)

# Calculate the 95% confidence interval
confidence_interval = norm.interval(
    0.95, loc=sensor_ZSBN_with_additional_data["CO2_predicted"], scale=std_error
)
sensor_ZSBN_with_additional_data["confidence_interval_lower"] = np.round(
    confidence_interval[0], decimals=4
)
sensor_ZSBN_with_additional_data["confidence_interval_upper"] = np.round(
    confidence_interval[1], decimals=4
)

# %%
# Create the interactive plot with actual CO2 measurements, predicted values and the 95% confidence interval
fig = go.Figure()

# Add trace for actual CO2 measurements
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN_with_additional_data["timestamp"],
        y=sensor_ZSBN_with_additional_data["CO2"],
        mode="lines",
        name="Actual CO2",
        hoverinfo="text",
        text=(
            "Actual CO2: "
            + np.round(sensor_ZSBN_with_additional_data["CO2"], decimals=4).astype(str)
        ),
    )
)

# Add trace for predicted CO2 values
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN_with_additional_data["timestamp"],
        y=sensor_ZSBN_with_additional_data["CO2_predicted"],
        mode="lines",
        name="Predicted CO2",
        hoverinfo="text",
        text=[
            f"Date: {date.strftime('%b %d, %Y %I:%M %p')}<br><br>Predicted CO2: {pred}<br>Upper CI: {upper}<br>Lower CI: {lower}"
            for date, upper, pred, lower in zip(
                pd.to_datetime(sensor_ZSBN_with_additional_data["timestamp"]),
                sensor_ZSBN_with_additional_data["CO2_predicted"],
                sensor_ZSBN_with_additional_data["confidence_interval_upper"],
                sensor_ZSBN_with_additional_data["confidence_interval_lower"],
            )
        ],
    )
)

# Add trace for the lower bound of the confidence interval
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN_with_additional_data["timestamp"],
        y=confidence_interval[0],
        line=dict(width=0),
        hoverinfo="none",
        showlegend=False,
    )
)

# Add trace for the upper bound of the confidence interval
fig.add_trace(
    go.Scatter(
        x=sensor_ZSBN_with_additional_data["timestamp"],
        y=confidence_interval[1],
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        hoverinfo="none",
        showlegend=False,
    )
)

# Add a vertical line for the start of drifting and an annotation for the drifting day
fig.add_vline(x=drifting_start, line_width=3, line_dash="dash", line_color="red")
fig.add_annotation(x=drifting_start, y=sensor_ZSBN_with_additional_data[sensor_ZSBN_with_additional_data['timestamp'] == drifting_start]['CO2'].values[0],
                   text="Drifting day", showarrow=True, arrowhead=1, ax=-50, ay=-100)

# Update the layout of the figure
fig.update_layout(
    title="CO2 Level Prediction with 95% Confidence Interval Using Similar Sensors Features",
    xaxis_title="Date",
    yaxis_title="CO2 Level (ppm)",
    hovermode="x unified",
    showlegend=True,
)

fig.update_xaxes(dtick=24 * 60 * 60 * 1000)

fig.show()


# %% [markdown]
# ### Observations and Findings: Analysis Report

# %% [markdown]
# The reduced size of the confidence intervals in the second plot speaks volumes about the improved precision of the model. Previously, while the model was adept at predicting the general daily trends in CO2 levels, it tended to smooth over the data, failing to capture sharp peaks and sudden fluctuations. This smoothing effect pointed to a model that, though somewhat effective in following the rhythm of daily changes, lacked sensitivity to the acute, momentary events that could cause significant CO2 spikes, likely due to traffic or other transient sources.
#
# Now, with the incorporation of data from similar sensors, the confidence intervals have narrowed considerably. This tightening of predicted values around the actual readings shows that the model is not just better at tracking the daily pattern—it's now finely attuned to the intricacies within the data, reflecting the actual behavior of the CO2 levels much more closely. 
#
# In essence, the model has become not just a mirror of average conditions but a detailed reflection of the sensor's environment, capable of picking up on the nuances and variations that are characteristic of an urban CO2 landscape. 
#
# This enhanced precision and the tightened confidence intervals boost our confidence in the model's predicted values, especially after the drifting day. Where previous predictions could be marred by uncertainty following anomalies, the model's improved ability to grasp acute fluctuations reinforces the reliability of its post-drift predictions.

# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c)
# - Leverage at least two different feature selection methods
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings

# %% [markdown]
# ### Helper function 

# %%
def get_fitted_dataframe(features: list):
    '''
    Helper function that fits a linear regression and returns a 
    dataframe with the predicted CO2 levels and the 95% confidence interval.
    '''

    X = (
        sensor_ZSBN_with_additional_data[
            sensor_ZSBN_with_additional_data.timestamp.dt.day < drifting_day
        ].loc[:,features]
    )
    
    y = sensor_ZSBN_with_additional_data[
        sensor_ZSBN_with_additional_data.timestamp.dt.day < drifting_day
    ][["CO2"]]

    # Fitting the model
    model = LinearRegression().fit(X, y)

    # Cross-validation using TimeSeriesSplit
    tscv = TimeSeriesSplit()
    scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")

    # Calculate root mean square errors
    rmse_errors = np.sqrt(-scores)

    # Standard deviation of the errors
    std_error = np.std(rmse_errors)

    # Predict the CO2 levels of the full dataset
    sensor_ZSBN_with_additional_data["CO2_predicted"] = np.round(
        model.predict(sensor_ZSBN_with_additional_data.loc[:,features]),
        decimals=4,
    )

    # Calculate the 95% confidence interval
    confidence_interval = norm.interval(
        0.95, loc=sensor_ZSBN_with_additional_data["CO2_predicted"], scale=std_error
    )
    sensor_ZSBN_with_additional_data["confidence_interval_lower"] = np.round(
        confidence_interval[0], decimals=4
    )
    sensor_ZSBN_with_additional_data["confidence_interval_upper"] = np.round(
        confidence_interval[1], decimals=4
)
    
    return sensor_ZSBN_with_additional_data


# %%
def plot_dataframe(dataframe, selected_features):
    
    # Create the interactive plot with actual CO2 measurements, predicted values and the 95% confidence interval
    fig = go.Figure()

    # Add trace for actual CO2 measurements
    fig.add_trace(
        go.Scatter(
            x=dataframe["timestamp"],
            y=dataframe["CO2"],
            mode="lines",
            name="Actual CO2",
            hoverinfo="text",
            text=(
                "Actual CO2: "
                + np.round(dataframe["CO2"], decimals=4).astype(str)
            ),
        )
    )

    # Add trace for predicted CO2 values
    fig.add_trace(
        go.Scatter(
            x=dataframe["timestamp"],
            y=dataframe["CO2_predicted"],
            mode="lines",
            name="Predicted CO2",
            hoverinfo="text",
            text=[
                f"Date: {date.strftime('%b %d, %Y %I:%M %p')}<br><br>Predicted CO2: {pred}<br>Upper CI: {upper}<br>Lower CI: {lower}"
                for date, upper, pred, lower in zip(
                    pd.to_datetime(dataframe["timestamp"]),
                    dataframe["CO2_predicted"],
                    dataframe["confidence_interval_upper"],
                    dataframe["confidence_interval_lower"],
                )
            ],
        )
    )

    # Add trace for the lower bound of the confidence interval
    fig.add_trace(
        go.Scatter(
            x=dataframe["timestamp"],
            y=dataframe["confidence_interval_lower"],
            line=dict(width=0),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Add a vertical line for the start of drifting and an annotation for the drifting day
    fig.add_vline(x=drifting_start, line_width=3, line_dash="dash", line_color="red")
    fig.add_annotation(x=drifting_start, y=dataframe[dataframe['timestamp'] == drifting_start]['CO2'].values[0],
                    text="Drifting day", showarrow=True, arrowhead=1, ax=-50, ay=-100)

    # Add trace for the upper bound of the confidence interval
    fig.add_trace(
        go.Scatter(
            x=dataframe["timestamp"],
            y=dataframe["confidence_interval_upper"],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Update the layout of the figure
    fig.update_layout(
        title=f"CO2 Level Prediction using features: {selected_features}",
        xaxis_title="Date",
        yaxis_title="CO2 Level (ppm)",
        hovermode="x unified",
        showlegend=True,
    )

    fig.update_xaxes(dtick=24 * 60 * 60 * 1000)

    fig.show()



# %% [markdown]
# ### __Method 1:  **Sequential Feature Selector**

# %% [markdown]
# In this section we used sequential feature selection. The method sequentially adds or removes features and greedily picks the estimators that are the best feature subset. It stops when the added benefit of adding the feature is smaller than a specified amount. The method allows us to also specify the number of features we'd like to select. We used this here to get two results: one with only the most important feature and one with the best mix of features. 

# %%
from sklearn.feature_selection import SequentialFeatureSelector

# # Sequential feature selection 
# # No restriction on the number of features to select
lr = LinearRegression()
sfs = SequentialFeatureSelector(lr, cv=tscv)
sfs.fit(X, y)

# Only select most impactful feature 
sfs_one = SequentialFeatureSelector(lr, direction='backward', n_features_to_select=1, cv=tscv)
sfs_one.fit(X, y)

# Create DataFrame for feature selection results
results_sfs = pd.DataFrame({
    'Feature': X.columns,
    'No restriction': [ 'Yes' if support else 'No' for support in sfs.get_support()],
    'Single Most Important Feature': [ 'Yes' if support_single else 'No' for support_single in sfs_one.get_support()]
})

display(results_sfs)
# Print selected features
selected_features = X.columns[sfs.get_support()].tolist()
print(f"\nThe features selected by this model are then: {selected_features}")

# Print selected most_important_feature
most_important_feature = X.columns[sfs_one.get_support()].tolist()
print(f"\nThe most important feature selected byt the model is: {most_important_feature}")

# %%
sequential_no_restriction = get_fitted_dataframe(selected_features)
plot_dataframe(sequential_no_restriction, selected_features)

# %%
sequential_best = get_fitted_dataframe(most_important_feature)
plot_dataframe(sequential_best, most_important_feature)

# %% [markdown]
# The model picks the CO2 level of the site ZHRG as the most important feature (second plot) this sensor is very close to the observed output of the  Actual CO2 levels, but it lacks in the observed spikes that are present for example on October 5th, 16th ,and 17th. Given that these sensors are probably in close proximity this is not surprising. 
#
# The best feature mix consists of the features of the CO2 level of sites ZHRG and ZHRZ and of the humidity recorded by the drifting sensor (first plot). Comparing the model predictions by eye indeed gives the impression that the feature mix is tracking actual CO2 levels more closely, although this model tends to struggle in the same places as the ZHRG_CO2-only model such as not tracking the peaks closely enough (e.g. Oct 5, 11, 16, 17, 19). 

# %% [markdown]
# ### __Method 2: **Lasso**

# %% [markdown]
# In this section we used the Lasso method to do feature selection. The Lasso regression introduces hyperparameters that punish coefficient weights. This form of regularization favors coefficient sparsity. 

# %%
from sklearn.linear_model import Lasso

# Lasso regression
lasso = Lasso(alpha=1)
lasso.fit(X, y)

# Store the coefficient in a Dataframe
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': np.round(lasso.coef_, 2)})
coefficients_df.sort_values(by='Coefficient', ascending=False, inplace=True)
display(coefficients_df)


# %%
# Select features with coefficients higher than 0.2
selected_features_lasso = coefficients_df[coefficients_df['Coefficient'] >= 0.2]['Feature'].tolist()
print(f"The features selected by the Lasso model are: {selected_features_lasso}")

lasso_dataframe = get_fitted_dataframe(selected_features_lasso)
plot_dataframe(lasso_dataframe, selected_features_lasso)

# %% [markdown]
# After running the Lasso method we see that it has eliminated the coefficients for the feature: temperature. The rest of the coefficients range from 0.38 to 0.1 in magnitude. Comparing with the feature-mix from method 1 we see that there is consensus on the importance of humidity and the CO2 level of site ZHRG, but Lasso seems to think that CO2 level of site ZUBG is the third most important feature, while in method 1 its was in the site ZHRZ although the coefficient weights are very similar. Based on the coefficients we decided to use all features that have a magnitude >= 0.2 and fit the predictions based on those features. 
#
# The CO2 predictions that stem from this feature set closely resemble the predictions from both method 1 and the full feature set. This is not surprising, as we have seen in method 1 that the CO2 level of site ZHRG is capturing already the bulk of the signal. Much like the predictions based on the other feature sets used this prediction also struggles to capture the sharp peaks of CO2 concentration during the morning hours, but captures the overall signal well. 

# %% [markdown]
# ### e) **5/35**
#
# Eventually, you'd like to try something new - __Bayesian Structural Time Series Modelling__ - to reconstruct counterfactual values, that is, what the CO2 measurements of the faulty sensor should have been, had the malfunction not happened on October 24. You will use:
# - the information of provided by similar sensors - the ones you identified in question c)
# - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
#
# To answer this question, you can choose between a Python port of the CausalImpact package (such as https://github.com/jamalsenouci/causalimpact) or the original R version (https://google.github.io/CausalImpact/CausalImpact.html) that you can run in your notebook via an R kernel (https://github.com/IRkernel/IRkernel).
#
# Before you start, watch first the [presentation](https://www.youtube.com/watch?v=GTgZfCltMm8) given by Kay Brodersen (one of the creators of the causal impact implementation in R), and this introductory [ipython notebook](https://github.com/jamalsenouci/causalimpact/blob/HEAD/GettingStarted.ipynb) with examples of how to use the python package.
#
# - Report your findings:
#     - Is the counterfactual reconstruction of CO2 measurements significantly different from the observed measurements?
#     - Can you try to explain the results?

# %%
from causalimpact import CausalImpact

pre_period = [
    0,
    sensor_ZSBN_with_additional_data.index[
        sensor_ZSBN_with_additional_data.timestamp.dt.day < drifting_day
    ].max(),
]
post_period = [
    sensor_ZSBN_with_additional_data.index[
        sensor_ZSBN_with_additional_data.timestamp.dt.day >= drifting_day
    ].min(),
    sensor_ZSBN_with_additional_data.index.max(),
]


# get features starting from the similar sensors finishing with "_CO2":
features = [str(similar_sensor) + "_CO2" for similar_sensor in similar_sensors] + [
    "humidity",
    "temperature",
]

data = sensor_ZSBN_with_additional_data[["CO2"] + features].rename(
    columns={"CO2": "y"} | {feature: f"x{i+1}" for i, feature in enumerate(features)}
)


impact = CausalImpact(data, pre_period, post_period)
impact.run()
impact.plot()

# %% [markdown]
# The important plots in this result for us are the first two plots. Plot 1 shows how the counterfactual model compares to the observed values. Plot 2 shows the differences to the counterfactual model, indicating to us clearly that the observed values are significantly lower than what we'd expect with the counterfactual model. Plot 3 would be important if the differences could be 'summed up'. If an advertiser would like to know how effective their online campain was, this plot would inform them about the total amount of extra clicks they got because of the campain. In our case the differences between the counterfactual model and the observed values are not summable, thus we can ignore this plot.
#
# In first two plots from the counterfactual reconstruction of the sensor data we see that the model shows a significant departure from the observed values for the first day. This is indicated by the observed values being outside the confidence intervals of the counterfactual model. At day 2 we start to fall inside the confidence interval. This is because the uncertainty of the model increases over time. Given that the first day is well outside the confidence interval it is plausible to say that the model is able to catch the drift. If we would have set up the model such that we get alarmed when a sensor starts drifting, the model would have alarmed us about this fact. 
# # That's all, folks!
