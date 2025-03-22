# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: PySpark
#     language: python
#     name: pysparkkernel
# ---

# %% [markdown]
# ---
# # DSLab Homework3 - Uncovering Public Transport Conditions using SBB data
#
# ## ... and learning about Spark `DataFrames` along the way
#
# In this notebook, we will use temporal information about sbb transports to discover delay trends. 
#
# ## Hand-in Instructions:
#
# - __Due: ~07.05.2024~ 14.05.2024 23:59:59 CET__
# - your project must be private
# - This notebook must be run in a PySpark kernel
# - `git push` your final verion to the master branch of your group's Renku repository before the due date
# - check if `Dockerfile`, `environment.yml` and `requirements.txt` are properly written
# - add necessary comments and discussion to make your codes readable
# - make sure the image builds successfully and your code is runnable

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-info">
# <b>ℹ️ Notes</b>:
# <ul>
# <li>
# Spark usage is not mandatory for answering every question unless explicitly stated but is highly encouraged. Feel free to split the work in separate notebooks if this is more convenient for you.
# </li>
# <li>
# When using Spark, you may answer the questions using DataFrame API, e.g. <em>spark.DataFrame.select...</em>, RDD api, or as pure SQL commands using <em>spark.DataFrame.sql(...)</em>. 
# </li>
# </ul>
# </div>

# %% [markdown]
# <div style="font-size: 100%" class="alert alert-block alert-info">
#     <b>ℹ️  Fair Cluster Usage:</b> As there are many of you working with the cluster, we encourage you to:
#     <ul>
#         <li>Whenever possible, prototype your queries on small data samples or partitions before running them on whole datasets</li>
#         <li>Save intermediate data in your HDFS home folder <b>f"/user/{username}"</b> (you may need hdfs dfs -setfacl)</li>
#         <li>Convert the data to an efficient storage format if needed</li>
#         <li>Use spark <em>cache()</em> and <em>persist()</em> methods wisely to reuse intermediate results</li>
#     </ul>
# </div>
#
# For instance:
#
# ```python
#     # Read a subset of the original dataset into a spark DataFrame
#     df_sample = spark.read.csv('/data/csv/', header=True).sample(0.01)
#     
#     # Save DataFrame sample
#     df_sample.mode("overwrite").write.orc(f'/user/{username}/sample.orc', mode='overwrite')
#
# ```

# %%
# %%configure -f
{ "conf": {
        "mapreduce.input.fileinputformat.input.dir.recursive": true,
        "spark.sql.extensions": "com.hortonworks.spark.sql.rule.Extensions",
        "spark.kryo.registrator": "com.qubole.spark.hiveacid.util.HiveAcidKyroRegistrator",
        "spark.sql.hive.hiveserver2.jdbc.url": "jdbc:hive2://iccluster065.iccluster.epfl.ch:2181,iccluster080.iccluster.epfl.ch:2181,iccluster066.iccluster.epfl.ch:2181/;serviceDiscoveryMode=zooKeeper;zooKeeperNamespace=hiveserver2",
        "spark.datasource.hive.warehouse.read.mode": "JDBC_CLUSTER",
        "spark.driver.extraClassPath": "/opt/cloudera/parcels/SPARK3/lib/hwc_for_spark3/hive-warehouse-connector-spark3-assembly-1.0.0.3.3.7190.2-1.jar",
        "spark.executor.extraClassPath": "/opt/cloudera/parcels/SPARK3/lib/hwc_for_spark3/hive-warehouse-connector-spark3-assembly-1.0.0.3.3.7190.2-1.jar"
    }
}

# %% [markdown]
# ---
# ## Start a spark Session environment

# %% [markdown]
# #### Session information (%%info)
#
# Livy is an open source REST server for Spark. When you execute a code cell in a PySpark notebook, it creates a Livy session to execute your code. You can use the %%info magic to display the current Livy sessions information, including sessions of others.

# %%
# %%info

# %% [markdown]
# Execute a python code that disply the Spark version in a cell, a new Spark session is started for this notebook if one does not exist yet.
#
# The links in this notebook all direct to the latest documentation, which may correspond to a more recent version of our Spark deployment. If necessary, you can replace 'latest' in the URL with the specific version number.

# %%
print(f'Start Spark name:{spark._sc.appName}, version:{spark.version}')

# %% [markdown]
# The new sessions is listed if you use the %%info magic function again

# %%
# %%info

# %% [markdown]
# A SparkSession object `spark` is initialized.

# %%
type(spark)

# %% [markdown]
# Be nice to others - remember to add a cell `spark.stop()` at the end of your notebook.

# %% [markdown]
# ---
# <div style="font-size: 100%" class="alert alert-block alert-info">
# <b>⚠️ Reminder!</b>
# <br>
# The default behavior in a PySpark notebook is to execute code remotely on the Spark cluster. However, if you employ the <em>%%local</em> spark magic, the code is executed locally instead.
# </div>

# %% [markdown]
# You can confirm this by executing a command that displays the value of the USER environment variable. Start by running it on the remote cluster (without using %%local), and then check it in the local environment as well.

# %%
import os
print(f"remote USER={os.getenv('USER',None)}")

# %%
# %%local
import os
print(f"local USER={os.getenv('USER',None)}")

# %% [markdown]
# Below, we provide the `username` and `hadoop_fs` as Python variables accessible in both environments. You can use them to enhance the portability of your code, as demonstrated in the following Spark SQL command. Additionally, it's worth noting that you can execute SQL commands on Hive directly from Spark.

# %%
# %%local
import os
username=os.getenv('USER', 'anonymous')
hadoop_fs=os.getenv('HADOOP_DEFAULT_FS', 'hdfs://iccluster067.iccluster.epfl.ch:8020')
print(f"local username={username}\nhadoop_fs={hadoop_fs}")

 # %%
 # (prevent deprecated np.bool error since numpy 1.24, until a new version of pandas/Spark fixes this)
import numpy as np
np.bool = np.bool_

username=spark.conf.get('spark.executorEnv.USERNAME', 'anonymous')
hadoop_fs=spark.conf.get('spark.executorEnv.HADOOP_DEFAULT_FS','hdfs://iccluster067.iccluster.epfl.ch:8020')
print(f"remote username={username}\nhadoop_fs={hadoop_fs}")

# %%
spark.sql(f'CREATE DATABASE IF NOT EXISTS {username}')
spark.sql(f'SHOW TABLES IN {username}').show(truncate=False)

# %% [markdown]
# 💡 You can convert spark DataFrames to pandas DataFrame to process the results. Only do this for small result sets, otherwise your spark driver will run OOM.

# %%
spark.sql(f'SHOW TABLES IN {username}').toPandas()

# %% [markdown]
# 💡 Alternatively, you can copy a variable from the Spark environment to your local context.
#
# `%%spark?`
# ```
# %spark [-o OUTPUT] [-m SAMPLEMETHOD] [-n MAXROWS] [-r SAMPLEFRACTION]
#              [-c COERCE]
#
# options:
#   -o OUTPUT, --output OUTPUT
#                         If present, indicated variable will be stored in
#                         variableof this name in user's local context.
#   -m SAMPLEMETHOD, --samplemethod SAMPLEMETHOD
#                         Sample method for dataframe: either take or sample
#   -n MAXROWS, --maxrows MAXROWS
#                         Maximum number of rows that will be pulled back from
#                         the dataframe on the server for storing
#   -r SAMPLEFRACTION, --samplefraction SAMPLEFRACTION
#                         Sample fraction for sampling from dataframe
#   -c COERCE, --coerce COERCE
#                         Whether to automatically coerce the types (default,
#                         pass True if being explicit) of the dataframe or not
#                         (pass False)
# ```
#
# For instance, to copy the complete (-n 1) variable data_frame:
# ```
# %%spark -o data_frame -n -1
# ```

# %% [markdown]
# ---
# ## PART I: First Steps with Spark DataFrames using Weather Data (20 points)
#
# We copied several years of historical weather data downloaded from [Wunderground](https://www.wunderground.com). Let's see if we can see any trends in this data. 

# %% [markdown]
# ### a) Load and inspect data - 1/20
#
# Load all the JSON data located in `/data/wunderground/json/history/` into a Spark DataFrame `df` using the appropriate method from the SparkSession.
#
# * Print the schema of the DataFrame.
#   * Note: Spark automatically detects the schema from the files, and the partitioning from the folder hierarchy; can you infer which columns are partitions?
# * Examine one row of the dataset, paying attention to the timestamps and their units.
#   * Note: Lay out the result vertically, one column per line.
#
# 💡 See: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html

# %%
# TODO - read wunderground json data from HDFS
df = spark.read.json('/data/wunderground/json/history/')

# %%
# TODO - print schema
df.printSchema()

# %%
# TODO - show one row
df.show(1, vertical=True)

# %% [markdown]
# ### b) User Defined and Built-in Functions - 1/20

# %%
import pyspark.sql.functions as F


# %% [markdown]
# __User-defined functions__
#
# A neat trick of spark dataframes is that you can essentially use something very much like an RDD `map` method but without switching to the RDD. If you are familiar with database languages, this works very much like e.g. a user-defined function in SQL. 
#
# So, for example, if we wanted to make a user-defined python function that returns a string value in lowercase, we could do something like this:

# %%
@F.udf
def lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


# %% [markdown]
# The `@F.udf` is a "decorator" -- this is really handy python syntactic sugar and in this case is equivalent to:
#
# ```python
# def lowercase(text):
#     return text.lower()
#     
# lowercase = F.udf(lowercase)
# ```
#
# It basically takes our function and adds to its functionality. In this case, it registers our function as a pyspark dataframe user-defined function (UDF).
#
# Using these UDFs is very straightforward and analogous to other Spark dataframe operations. For example:

# %%
df.select(df.site,lowercase(df.site).alias('lowercase_site')).show(n=5)

# %% [markdown]
# __Built-in functions__
#
# Using a framework like Spark is all about understanding the ins and outs of how it functions and knowing what it offers. One of the cool things about the dataframe API is that many functions are already defined for you (turning strings into lowercase being one of them).
#
# Spark provides us with some handy built-in dataframe functions that are made for transforming date and time fields.
#
# Find the [Spark python API documentation](https://spark.apache.org/docs/latest/api/python/index.html). Look for the `sql` section and find the listing of `pyspark.sql.functions`. Using built-in functions, convert the GMT valid_time_gmt to YYYY-mm-dd HH:MM:SS format and display 5 rows to verify the results.
#
# The output should be similar to:
#
# ```
# +--------------+-------------------+
# |valid_time_gmt|               date|
# +--------------+-------------------+
# |    1672528800|2023-01-01 00:20:00|
# |    1672530600|2023-01-01 00:50:00|
# |    1672532400|2023-01-01 01:20:00|
# |    1672534200|2023-01-01 01:50:00|
# |    1672536000|2023-01-01 02:20:00|
# +--------------+-------------------+
# ```

# %%
# TODO convert valid_time_gmt
df.select(df.valid_time_gmt, F.from_unixtime(df.valid_time_gmt).alias('date')).show(5)

# %% [markdown]
# We'll work with a combination of these built-in functions and user-defined functions for the remainder of this homework. 
#
# Note that the functions can be combined. Consider the following dataframe and its transformation:

# %% [markdown]
# ```
# from pyspark.sql import Row
#
# # create a sample dataframe with one column "degrees" going from 0 to 180
# test_df = spark.createDataFrame(spark.sparkContext.range(180).map(lambda x: Row(degrees=x)), ['degrees'])
#
# # define a function "sin_rad" that first converts degrees to radians and then takes the sine using built-in functions
# sin_rad = F.sin(F.radians(test_df.degrees))
#
# # show the result
# test_df.select(sin_rad).show()
# ```

# %% [markdown]
# ### c) Transform the data - 4/20
#
# Now, let's see how we can start to organize the weather data by their timestamps. Remember, our goal is to weather trends on a timescale of hours. A much needed column then is simply `dayofmonth`, and `hour`.
#
#
# Try to match this view:
#
# Create a dataframe called `weather_df` that includes the columns `month`, `dayofmonth`, `hour` and `minutes`, calculated from `valid_time_gmt`
# - It contains all (and only) the data since 2022-01-01 00:00:00.
# - It contains other useful columns from the original data as show in the example below
# - Show the result
# - The first row (shown sorted in chronological order) should ressemble:
#
# ```
#  valid_time_gmt | 1640991600    
#  clds           | null          
#  day_ind        | N             
#  dewPt          | -2            
#  feels_like     | 4             
#  gust           | null          
#  heat_index     | 4             
#  obs_name       | Chateau-D'Oex 
#  precip_hrly    | 0.0           
#  precip_total   | null          
#  pressure       | 910.8         
#  rh             | 64            
#  temp           | 4             
#  uv_desc        | Low           
#  uv_index       | 0             
#  vis            | null          
#  wc             | 4             
#  wdir           | 100           
#  wdir_cardinal  | E             
#  wspd           | 2             
#  wx_phrase      | null          
#  site           | LSTS          
#  year           | 2022          
#  month          | 1             
#  dayofmonth     | 1             
#  hour           | 0             
#  minute         | 0  
# ```
#
# __Note:__ 
# - [pyspark.sql.DataFrame](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)

# %%
# TODO create weather_df as indicated above; extract all the necessary columns
weather_df = df.select(df.valid_time_gmt, df.clds, df.day_ind, df.dewPt, df.feels_like, df.gust, 
                       df.heat_index, df.obs_name, df.precip_hrly, df.precip_total, df.pressure, 
                       df.rh, df.temp, df.uv_desc, df.uv_index, df.vis, df.wc, df.wdir, df.wdir_cardinal,
                       df.wspd, df.wx_phrase, df.site, 
                       F.year(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('year'),
                       F.month(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('month'),
                       F.dayofmonth(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('dayofmonth'),
                       F.hour(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('hour'),
                       F.minute(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('minute'))


# %% [markdown]
# Optional: you may save the file to HDFS to avoid recomputation when experimenting with this notebook. However, it's important to note that due to Spark partitioning, the original order may not be preserved when reading the data back from the file.

# %%
# Filter the data to display everything from 2022 onwards and to sort chronologically
weather_df = weather_df.filter(weather_df.year >= 2022).sort(df.valid_time_gmt, ascending=True)

# %%
weather_df.show(1, vertical=True)

# %% [markdown]
# ### d) Top average monthly precipitation per site - 4/20
#
# We used `groupBy` already in the exercises notebooks, but here we will take more advantage of its features. 
#
# One important thing to note is that unlike other RDD or DataFrame transformations, the `groupBy` does not return another DataFrame, but a `GroupedData` object instead, with its own methods. These methods allow you to do various transformations and aggregations on the data of the grouped rows. 
#
# Conceptually the procedure is a lot like this:
#
# ![groupby](https://i.stack.imgur.com/sgCn1.jpg)
#
# The column that is used for the `groupBy` is the `key` - it can be a list of column keys, e.g. groupby('key1','key2',...) - once we have the values of a particular key all together, we can use various aggregation functions on them to generate a transformed dataset. In this example, the aggregation function is a simple `sum`.

# %% [markdown]
# **Question:**
#
# Starting from `weather_df`, use group by, and aggregations to show the 10 top (site,month of year) with the highest total precipitation for the month, averaged over 2022-2023.
#
# Name the spark DataFrame `avg_monthly_precip_df`.
#
# The schema of the table is:
# ```
# root
#  |-- site: string (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- avg_total_precip: double (nullable = true)
# ```
#
# Note:
# * A site may report multiple hourly precipitation measurements (precip_hrly) within a single hour. To prevent adding up hourly measurement for the same hour, you should compute the max values observed at each site within the same hour.
# * Some weather stations do not report the hourly  precipitation, they will be shown as _(null)_
# * Weather_df contains historical weather data starting 1.1.2022 up to very recently - we are only asking for measurements from 2022 to 2023.
#
# **Checkpoint**: at the top of the list, we recorded a maximum avg precipitation of 228.5 mm.

# %%
# We will first drop all the rows that have null as precipitation for hour 
# and filter to keep only data from 2022 and 2023
tt = weather_df.filter(~F.isnull(weather_df.precip_hrly)) \
               .filter((weather_df.year == 2022) | (weather_df.year == 2023))   
tt = tt.select(tt.site, tt.precip_hrly, tt.year, tt.month, tt.dayofmonth, tt.hour)

# %%
# First, we will keep only the maximum precipitation value for the same site within the same hour
tt = tt.groupBy([tt.site, tt.year, tt.month, tt.dayofmonth, tt.hour]) \
       .agg(F.max(tt.precip_hrly).alias('max_precip_hrly'))

# %%
# Then, we will sum all the values that belong to the same site and month
tt = tt.groupBy([tt.site, tt.year, tt.month]) \
       .agg(F.sum(tt.max_precip_hrly).alias('total_precip_month'))

# %%
# And now average over months in these 2 years
avg_monthly_precip_df = tt.groupBy([tt.site, tt.month]) \
                           .agg(F.avg(tt.total_precip_month).alias('avg_total_precip'))

# %%
avg_monthly_precip_df.printSchema()

# %%
# TODO: show first 10 (site,month,avg_total_precip) samples sorted by precipitation in deacreasing order
avg_monthly_precip_df = avg_monthly_precip_df.sort(avg_monthly_precip_df.avg_total_precip, ascending=False)
avg_monthly_precip_df.show(10)

# %%
avg_monthly_precip_df.coalesce(1).write.parquet(f'/user/{username}/avg_monthly_precip', mode="overwrite")
avg_monthly_precip_df = spark.read.parquet(f'/user/{username}/avg_monthly_precip')
avg_monthly_precip_df.printSchema()

# %% [markdown]
# 💡 You can copy DataFrames and byte strings from the Spark cluster and visualize them locally in the notebook.

# %% magic_args="-c False -o avg_monthly_precip_df -n -1" language="spark"
#

# %%
weather_df.select("site").distinct().count()

# %%
avg_monthly_precip_df.select("site").distinct().count()

# %%
# %%local
avg_monthly_precip_df.dropna().sort_values('month',ascending=True)

# %% [markdown]
# 💡 Or you can use matplotlib or pandas.plot on the remote cluster, noting that:
# 1. Matplotlib should be set to a [static backend](https://matplotlib.org/stable/users/explain/figure/backends.html), because it is generated remotely (not interactive)
# 2. Clear or close the figure, to refresh the figure and prevent OOM on the driver side

# %%
import matplotlib.pyplot as plt
plt.switch_backend('agg') #[1] - optional, this should be the default.
plt.close() #[2]

## Convert to pandas, then do normal pandas ops
# - drop NaN
# - Pivot (long to wide) table using month as index (horizontal x axis) and moving each distinct 'site' to a separate column
df = avg_monthly_precip_df.toPandas().dropna()

# Switch between True and False to try different visualizations
if False:
    ax=df.pivot(index='month',columns=['site']).plot.bar(y='avg_total_precip',figsize=(16,6),grid=True)
    ax.set_title('Monthly precipitations averaged over 2022-2023 per site')
    ax.legend(loc='lower right',bbox_to_anchor=(0.5, 0., 0.5, 0.5), ncols=4)
else:
    #ddf=df[['month','avg_total_precip']].groupby('month').agg(pavg=('avg_total_precip','mean'),pmin=('avg_total_precip','min'),pmax=('avg_total_precip','max'))
    #plt.errorbar(x=ddf.index,y=ddf.pavg,yerr=ddf[['perc']].values.T,fmt='o')
    ddf=df[['month','avg_total_precip']].groupby('month').quantile(q=[0.1,0.5,0.9])
    q=ddf.values.reshape(3,12)
    plt.errorbar(x=ddf.index.get_level_values(0).unique().tolist(),y=q[1],yerr=[q[0],q[2]],fmt='o')
    plt.title('[.1 .5 .9] quantiles of monthly precip over 22-23 for all sites')
    plt.figsize=(16,6)
df=None
ddf=None

# %%
# %matplot plt

# %% [markdown]
# ### e) Spark Windows  - 5/20
#
# Window functions are another awesome feature of dataframes. They allow users to accomplish complex tasks using very concise and simple code. 
#
# Above we computed just the total average montly precipitation for *any* site.
#
# Now lets say that each day we want to know the site with the highest reported temperature.
#
# This is a non-trivial thing to compute and requires "windowing" our data. We recommend reading this [window functions article](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html), the [spark.sql.Window](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)  and optionally the [Spark SQL](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html) documentation to get acquainted with the idea. You can think of a window function as a fine-grained and more flexible `groupBy`. 
#
# There are two things we need to define to use window functions:
#
# 1. the "window" to use, based on which columns (partitioning) and how the rows should be ordered 
# 2. the computation to carry out for each windowed group, e.g. a max, an average etc.
#
# Lets see how this works in the next question. We will define a window function, `hourly_window` that will partition data based on the _(year,month,dayofmonth,hour)_ columns. Within each window, the rows will be ordered by the hourly temperature from highest to lowest. Finally, we will use the rank function **over** this window to give us the ranking of the sites with the highest temperatures.  We will then filter out the results to keep only the highest ranking sites.
#
# In the end, this is a fairly complicated operation achieved in just a few lines of code! (can you think of how to do this with an RDD??).

# %%
from pyspark.sql import Window

# %% [markdown]
# (a) First, define a _pyspark.sql.window.WindowSpec_ to specify the window partition and the ordering. We create a window definition that partitions the data based on the (_year_, _month_, _dayofmonth_, _hour_) columns and order all the rows (i.e. all the site measurements) in each partition by temperature (_temp_), in _decreasing_ order. As explained in the above article, it should follow the pattern:
#
# ```
# Window.partitionBy(...).orderBy(...)
# ```

# %%
# TODO
hourly_window = Window.partitionBy('year', 'month', 'dayofmonth', 'hour').orderBy(F.desc('temp'))

# %% [markdown]
# (b) Next you need to define a computation on _hourly_window_ that we want to order by in decreasing order. It is a window aggregation of type _pyspark.sql.column.Column_.
#
# Calculate the hourly ranking of temperatures. Use the helpful built-in F.rank() _spark.sql.function_, and call the _over_ method to apply it over the _hourly_window_, and name it (alias) _rank_.

# %%
# TODO
hourly_rank = F.rank().over(hourly_window).alias('rank')

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `RANK() OVER (PARTITION BY year, month, dayofmonth, hour ORDER BY temp DESC NULLS LAST ?) AS rank`, where the input data frame serves as the placeholder for the '?' symbol. This _window function_ assigns a rank to each record based on the specified partition and ordering criteria of _hourly_window_.

# %% [markdown]
# (c) Finally, apply `hourly_rank` to the `weather_df` DataFrame computed earlier.
#
# Filter the results to show all and only the sites with the 5 highest temperature per hour (if multiple sites have the same temperature, they count as one), then order the hourly measurements in chronological order, showing the top ranked sites in their ranking order.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +----+----+-----+----------+----+----+----+
# |site|year|month|dayofmonth|hour|temp|rank|
# +----+----+-----+----------+----+----+----+
# |LSTO|2022|    1|         1|   0|  10|   1|
# |LSPA|2022|    1|         1|   0|   9|   2|
# |LSGT|2022|    1|         1|   0|   8|   3|
# |LSGL|2022|    1|         1|   0|   7|   4|
# |LSZJ|2022|    1|         1|   0|   7|   4|
# |LSGB|2022|    1|         1|   0|   7|   4|
# |LSTO|2022|    1|         1|   1|  10|   1|
# (...)
# +----+----+-----+----------+----+----+----+
# ```

# %%
# TODO apply window to weather_df and show output, sorted as shown in example above
# Apply window function
window_df = weather_df.withColumn('rank', hourly_rank)
# Select columns that should be displayed
window_df = window_df.select('valid_time_gmt', 'site', 'year', 'month', 'dayofmonth', 'hour', 'temp', 'rank')

window_df.coalesce(1).write.parquet(f'/user/{username}/window_temp_site', mode="overwrite")

# Filter only ones that have rank less or equal to 5 and then sort chronologically
window_df = window_df.filter(window_df.rank <= 5).sort('valid_time_gmt', 'rank', ascending=True)
# Drop unnecessary column
window_df = window_df.drop('valid_time_gmt')
window_df.show(20)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### f) Sliding Spark Windows - 5/20

# %% [markdown]
# With window functions, you can also compute aggregate functions over a sliding window. 
#
# **Question:** calculate the average temperature temperature at each site over 3-hour sliding window.
#
# The process follows a similar pattern to the previous steps, with a few distinctions:
#
# * Rows are processed independently for each site.
# * The window slides over the timestamps (_valid_time_gmt_) in chronological order, spanning intervals ranging from 2 hours and 59 minutes (10740 seconds) before the current row's timestamp up to the current row's timestamp."
#
# 💡 [spark.sql.Window(Spec)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)

# %% [markdown]
# (a) First, define a _pyspark.sql.window.WindowSpec_ to specify the window partition, the row ordering inside the partition, and the _range_.

# %%
# TODO
sliding_3hour_window = Window.partitionBy('site').orderBy('valid_time_gmt')
sliding_3hour_window = sliding_3hour_window.rangeBetween(-10740, Window.currentRow)

# %% [markdown]
# (b) Next you need to define a computation on _rolling_3hour_window_. It is a window aggregation of type pyspark.sql.column.Column.
#
# Calculate the average ranking of temperatures. Use the helpful built-in F.avg() spark.sql.function, and call the over method to apply it over the _rolling_3hour_window_, and name it (alias) _avg_temp_.

# %%
# TODO
sliding_3hour_avg = F.avg('temp').over(sliding_3hour_window).alias('avg_temp')

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `avg(temp) OVER (PARTITION BY site ORDER BY valid_time_gmt ASC NULLS FIRST RANGE BETWEEN -10740 FOLLOWING AND CURRENT ROW) AS avg_temp`

# %%
print(sliding_3hour_avg)

# %% [markdown]
# (c) Finally, apply _sliding_3hour_avg_ to the _weather_df_ DataFrame computed earlier, and order chronologically. _Then_ filter the output to show the outpout of sites 'LSTO', 'LSZH and 'LSGL'.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +--------------+----+-----+----------+----+----+----+------------------+
# |valid_time_gmt|year|month|dayofmonth|hour|site|temp|          avg_temp|
# +--------------+----+-----+----------+----+----+----+------------------+
# |    1640991600|2022|    1|         1|   0|LSGL|   7|               7.0|
# |    1640991600|2022|    1|         1|   0|LSTO|  10|              10.0|
# |    1640992800|2022|    1|         1|   0|LSZH|   2|               2.0|
# |    1640994600|2022|    1|         1|   0|LSZH|   3|               2.5|
# |    1640995200|2022|    1|         1|   1|LSGL|   6|               6.5|
# |    1640995200|2022|    1|         1|   1|LSTO|  10|              10.0|
# |    1640996400|2022|    1|         1|   1|LSZH|   3|2.6666666666666665|
# |    1640998200|2022|    1|         1|   1|LSZH|   3|              2.75|
# (...)
# ```

# %%
# TODO - apply window to weather_df and show output, sorted as shown in example above
sld_window = weather_df.select('valid_time_gmt', 'year', 'month', 'dayofmonth', 'hour', 'site', 'temp')
sld_window = sld_window.withColumn('avg_temp', sliding_3hour_avg)

# %%
# Order chronologically
sld_window = sld_window.sort('valid_time_gmt', ascending=True)

# %%
# Filter and display
sld_window = sld_window.filter((sld_window.site=='LSTO') | (sld_window.site=='LSZH') | (sld_window.site=='LSGL'))
sld_window.show(10)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Bonus) Putting it all together - +1pt
#
# Can you combine _sliding_3hour_avg_ with _hourly_rank_? in order to show the weather stations with the 5 top temperatures averaged over the 3h sliding window?
#
# Hint: you can create a new _hourly_rank_ called _hourly_rank_3h_, or you can reuse _hourly_rank_ unchanged if you rename a column before applying it.

# %%
# TODO
# We first get table with avg_temp
bonus = weather_df.select('valid_time_gmt', 'year', 'month', 'dayofmonth', 'hour', 'site', 'temp')
bonus = bonus.withColumn('avg_temp', sliding_3hour_avg)

# %%
# We change hourly_rank to work on avg_temp
hourly_window_bonus = Window.partitionBy('year', 'month', 'dayofmonth', 'hour').orderBy(F.desc('avg_temp'))
hourly_rank_bonus = F.rank().over(hourly_window_bonus).alias('rank')

# %%
# Apply window function
window_df_bonus = bonus.withColumn('rank', hourly_rank_bonus)

# %%
# Select columns that should be displayed
window_df_bonus = window_df_bonus.drop('temp')

# %%
# Filter only ones that have rank less or equal to 5 and then sort chronologically
window_df_bonus = window_df_bonus.filter(window_df_bonus.rank <= 5).sort('valid_time_gmt', 'rank', ascending=True)
# Drop unnecessary column
window_df_bonus = window_df_bonus.drop('valid_time_gmt')
window_df_bonus.show(10)

# %% [markdown]
# ---

# %% [markdown]
# ## PART II: SBB Network - Clustering (20 points)
#
#
# ![graph](./figs/graph_clustering.svg)
#
# In this section we will try to do a slightly more complicated analysis of the SBB timetable data.
#
# Our objective is to cluster SBB stops according to their connectivity level, measured by the frequency of public transport connections between pairs of stops.
#
# To this end we will make use of the graph clustering methods [PowerIterationClustering](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.PowerIterationClustering.html#pyspark.ml.clustering.PowerIterationClustering) method, described in [Lin and Cohen, Power Iteration Clustering](http://www.cs.cmu.edu/~frank/papers/icml2010-pic-final.pdf).
#
# Graph clustering algorithms are powerful instruments for revealing patterns, communities, and structures within intricate networks, spanning from social networks and biological systems to recommendation engines[[1](https://memgraph.com/blog/graph-clustering-algorithms-usage-comparison)], and in our case, regions connected by public transports.
#
# * **Q:** What about Spark GraphX? **A:** do you want to learn Scala?
# * **Q:** What about [GraphFrames](https://graphframes.github.io/graphframes/docs/_site/index.html)? **A:** It is available on the cluster, and you can use it (import graphframes). Note however, what will become of it and its API stability remain [uncertain](https://github.com/graphframes/graphframes/issues/435).

# %%
import numpy as np
np.bool = np.bool_

username=spark.conf.get('spark.executorEnv.USERNAME', 'anonymous')
hadoop_fs=spark.conf.get('spark.executorEnv.HADOOP_DEFAULT_FS','hdfs://iccluster067.iccluster.epfl.ch:8020')
print(f"remote username={username}\nhadoop_fs={hadoop_fs}")

# %%
# %%local
import os
username=os.getenv('USER', 'anonymous')
hadoop_fs=os.getenv('HADOOP_DEFAULT_FS', 'hdfs://iccluster067.iccluster.epfl.ch:8020')
print(f"local username={username}\nhadoop_fs={hadoop_fs}")

# %% [markdown]
# ### a) Load and inspect data - 1/20
#
# ⚠️ The question in this section can be computationally demanding. It is advisable to begin by experimenting with smaller datasets first, such as those used in homework 2. If needed, you can access the files on HDFS under /data/sbb/share. Starting with smaller datasets enables faster iterations and helps to understand the computational requirements before moving on to larger datasets.
#
# * Read the stop-times data from `/data/sbb/parquet/timetables/stop_times` into a pyspark DataFrame (e.g. _stop_times_df_)

# %%
#TODO
stop_times_df = spark.read.parquet('/data/sbb/parquet/timetables/stop_times')

# %%
# TODO - explore data
stop_times_df.printSchema()
stop_times_df.show(1, vertical=True)

# %%
# Note the DataFrame.describe method can be slow
#stop_times_df.describe().show()

# %% [markdown]
# ### b) Transform the data - 8/20
#
# Create DataFrame `network_df` from `stop_times_df`, such that it has the following schema:
#
# ```
# root
#  |-- src: integer
#  |-- dst: integer
#  |-- weight
# ```
#
# Where:
# * `src` and `dst` the are two distinct stops connected by a same trip_id.
# * `weight` is a value that express the strength of the connection between the stops. You are free to explore different metrics. Start with a simple metric such as trip frequency between the stops, or difference between their stop sequences. Then experiment with other metrics if time permits.
# * If you specify _(src,dst,weight)_ then it is assumed that _(src,dst,weight) = (dst,src,weight)_ - you must only specify one direction.
#
# <ul style="list-style-type:none;">
# <li>💡 Take advantage of Spark MLlib API and helper classes.</li>
# <li>💡 Stop_ID are of type string. You must conver them to numerical indices before using them in the clustering model. Also make sure that the indices of src and dst are consistent, i.e. the numerical value of a same stop_id must be the same in src and dst.</li>
# </ul>
#
# See: [StringIndexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer), [IndexToString](https://spark.apache.org/docs/latest/ml-features.html#indextostring), and pyspark MLlib Scalers or Normalization methods.

# %%
from pyspark.ml.feature import StringIndexer, IndexToString

# %%
import pyspark.sql.functions as F

# %% [markdown]
# ### 1. Convert stop_times stop_id to indices 

# %%
# TODO - Convert stop_times stop_id to indices 

# Init the StringIndexer
string_indexer = StringIndexer(inputCol="stop_id", outputCol="stop_index")

# Fit the StringIndexer
string_indexer_model = string_indexer.fit(stop_times_df)

# create stop_trip_indexed from stop_trip_df
stop_trip_indexed = string_indexer_model.transform(stop_times_df)
stop_trip_indexed.show(1, vertical=True)

# %% [markdown]
# ### 2.Creation of network_df

# %%
# TODO: create network_df (hint: cast stop indices to integers)
network_df = stop_trip_indexed.alias("src") \
    .join(stop_trip_indexed.alias("dst"),
          (F.col("src.trip_id") == F.col("dst.trip_id")) &
          (F.col("src.stop_sequence") < F.col("dst.stop_sequence"))) \
    .select(F.col("src.stop_index").cast("int").alias("src"),
            F.col("dst.stop_index").cast("int").alias("dst"),
            F.col("src.trip_id").alias("trip_id"),
            F.col("src.stop_sequence").cast("int").alias("src_stop_sequence"),
            F.col("dst.stop_sequence").cast("int").alias("dst_stop_sequence"))

network_df.show(5)

# %% [markdown]
# ### 3. Exploration of weight metrics

# %% [markdown]
# #### a. Trip Frequency 

# %%
# Group network_df by source and destination stops and count the number of trips
trip_frequency_df = (network_df
                     .groupBy("src", "dst")
                     .agg(F.count(network_df.trip_id).alias("trip_frequency")))

trip_frequency_df.show(5)

# %%
# import matplotlib.pyplot as plt

# trip_frequency_pd = trip_frequency_df.toPandas()

# # Repartition of the frequency
# plt.figure(figsize=(10, 6))
# plt.hist(trip_frequency_pd['trip_frequency'], bins=20, color='skyblue', edgecolor='black')
# plt.xlabel('Trip Frequency')
# plt.ylabel('Count')
# plt.title('Distribution of Trip Frequency')
# plt.grid(True)
# plt.show()


# %%
# %matplot plt

# %% [markdown]
# #### b. Differences between stop sequences 

# %%
# Calculate the difference in stop sequence for each source and destination pair
diff_dist_df = (network_df
                .groupBy("src", "dst", "src_stop_sequence", "dst_stop_sequence")
                .agg(F.abs(network_df.dst_stop_sequence - network_df.src_stop_sequence).alias("stop_sequence_difference")))

diff_dist_df.show(5)

# %%
# Merge the stop frequency from trip_frequency_df and the stop distance from diff_dist_df to weigh the weights (the further they are, the less they weigh)
weighted_network_df = trip_frequency_df.join(diff_dist_df, ["src", "dst"])

weighted_network_df.show(5)

# %% [markdown]
# #### Normalization of `trip_frequency` and `stop_sequence_difference` using the min-max normalization formula

# %%
# Calculate min and max for trip_frequency and stop_sequence_difference
min_max_values = weighted_network_df.select(
    F.min("trip_frequency").alias("min_frequency"),
    F.max("trip_frequency").alias("max_frequency"),
    F.min("stop_sequence_difference").alias("min_stop_diff"),
    F.max("stop_sequence_difference").alias("max_stop_diff")
).collect()[0]

# %%
min_max_values

# %%
# Calculate normalized values for trip_frequency and stop_sequence_difference
weighted_network_df = weighted_network_df.withColumn(
    "normalized_frequency",
    (weighted_network_df.trip_frequency - min_max_values.min_frequency) / (min_max_values.max_frequency - min_max_values.min_frequency)
).withColumn(
    "normalized_stop_diff",
    (weighted_network_df.stop_sequence_difference - min_max_values.min_stop_diff) / (min_max_values.max_stop_diff - min_max_values.min_stop_diff)
)

weighted_network_df.show(1, vertical = True)

# %%
# Define a small epsilon value to prevent dividing by zero
epsilon = 1e-10

# Calculate a weighted weight based on trip frequency and distance in terms of stop sequence
weighted_weight_df = weighted_network_df.withColumn('weighted_weight', 
                                    weighted_network_df.normalized_frequency / (weighted_network_df.normalized_stop_diff + epsilon))

weighted_weight_df.show(1, vertical = True)

# %% [markdown]
# ### ADD a comment for explaining why we select this method and not jsut the frequency 

# %%
# Select the columns of interest from the calculated DataFrame
network_df = weighted_weight_df.select(weighted_weight_df.src,
                                       weighted_weight_df.dst,
                                       (weighted_weight_df.weighted_weight).alias('weight'))
network_df.show(5)

# %%
network_df.printSchema()

# %% [markdown]
# #### save the network in my database

# %%
network_df.coalesce(1).write.parquet(f'/user/{username}/network_part_II', mode="overwrite")

# %% [markdown]
# #### load the network in my database

# %%
network_df = spark.read.parquet(f'/user/{username}/network_part_II')
network_df.printSchema()
network_df.show(1, vertical = True)

# %% [markdown]
# ### c) Cluster the stops using PowerIterationClusters - 8/20
#
# Apply the [PowerIterationClustering](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.PowerIterationClustering.html)
# on _network_indexed_ to cluster the data.
#
# For an effective graph clustering, try to maximize a connectivity ratio that measures the strength of connections within clusters relative to connections between clusters. 
#
# Devise a method that finds  a reasonable $k$ value for the number of clusters using a minimum of steps. You can start from a reasonable number $k$, for instance the number of major cities in Switzerland and move from there, and increase $k$ until further increasing the number of clusters does not significantly improve the connectivity ratio.

# %% [markdown]
# #### max iter different 
# #### initMode 

# %%
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import PowerIterationClustering
import matplotlib.pyplot as plt

# Initialize lists to store k values and corresponding silhouette scores
k_values = []
silhouette_scores = []

# Initialize variable to store previous silhouette score
prev_silhouette = None

# Loop over k values from 8 to 20 with a step of 2
# for k in range(8, 21, 1):
for k in range(14, 15, 1):
    print(k)
    # Apply Clustering method on network_df
    pic = PowerIterationClustering(k=k, maxIter=10)
    assignments = pic.assignClusters(network_df)
    assignments = network_df.join(assignments, on=[(network_df.src == assignments.id)]).drop('id')
    assignments = assignments.withColumnRenamed('weight', 'features')

    # Create a VectorAssembler
    vecAssembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")

    # Transform the DataFrame
    assignments = vecAssembler.transform(assignments)

    # Rename 'cluster' to 'prediction'
    assignments = assignments.withColumnRenamed('cluster', 'prediction')

    # Use the vectorized features for evaluation
    evaluator = ClusteringEvaluator(featuresCol="features_vec")
    silhouette = evaluator.evaluate(assignments)

    # Store k and silhouette score
    k_values.append(k)
    silhouette_scores.append(silhouette)
    print("k={}, silhouette_score={}".format(k,silhouette))

    # Check if silhouette score has improved
    if prev_silhouette is not None and silhouette <= prev_silhouette :
        break

    # Update previous silhouette score
    prev_silhouette = silhouette

# Plot silhouette scores as a function of k
plt.close('all')
plt.plot(k_values, silhouette_scores)
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score as a function of k')
plt.show()

# %%
# %matplot plt

# %% [markdown]
# #### save the assigment in my database

# %%
assignments.coalesce(1).write.parquet(f'/user/{username}/assignments_part_II', mode="overwrite")

# %% [markdown]
# #### load the network in my database

# %%
assignments = spark.read.parquet(f'/user/{username}/assignments_part_II')
assignments.printSchema()
assignments.show(1, vertical = True)

# %%
prediction_distribution_df = (assignments.groupBy('prediction')
                                   .count()
                                   .orderBy('count', ascending=False)
                                   .toPandas())

# Afficher la répartition des prédictions
print(prediction_distribution_df)

# Afficher un histogramme
plt.close('all')
plt.figure(figsize=(10, 6))
plt.bar(prediction_distribution_df['prediction'], prediction_distribution_df['count'], color='skyblue')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.title('Prediction Distribution')
plt.show()

# %%
# %matplot plt

# %% [markdown]
# ### d) Visualize the stops locally in a map - 1/20

# %% [markdown]
# Visualize the clusters created in the previous question locally (%%local or in a separate python notebook).
#
# We recommend that you export the data to your local environment on JupyterHub, and save it in a file.
#
# Before copying the data to your local environment you will need to convert the stop indices back to their original stop_id and join (stop_id,cluster_id) with the stop locations from the stop tables in `/data/sbb/orc/timetables/stops` (using the SBB timetables published on 10.1.2024).

# %%
assignments.printSchema()

# %%
# TODO - inverse stop indices to stop names

# Initiate IndexToString for converting stop indices to stop names for both src and dst columns
index_to_string_src = IndexToString(inputCol="src", outputCol="stop_id_src").setLabels(string_indexer_model.labels)

# Apply transformations to convert stop indices to stop names for src and dst columns
assignments_with_names = index_to_string_src.transform(assignments)


assignments_with_names.printSchema()
assignments_with_names.show(1, vertical = True)

# %%
assignments_with_names.coalesce(1).write.parquet(f'/user/{username}/assignments_names_part_II', mode="overwrite")
assignments_with_names = spark.read.parquet(f'/user/{username}/assignments_names_part_II')
assignments_with_names.printSchema()
assignments_with_names.show(1, vertical = True)

# %%
# TODO - Joint with stop coordinates from /data/sbb/parquet/timetables/stops/ (hint: use timetable published on 2024-1-10)

# %%
stops_df = spark.read.orc('/data/sbb/orc/timetables/stops')


# %%
stops_df.show(20)

# %%
stop_cluster_loc_df = assignments_with_names.join(stops_df, assignments_with_names.stop_id_src == stops_df.stop_id).select('stop_id', 'stop_lat', 'stop_lon', 'prediction')
stop_cluster_loc_df.show()


# %%
stop_cluster_loc_df.coalesce(1).write.parquet(f'/user/{username}/stop_cluster_loc', mode="overwrite")
stop_cluster_loc_df = spark.read.parquet(f'/user/{username}/stop_cluster_loc')
stop_cluster_loc_df.printSchema()
stop_cluster_loc_df.show(1, vertical = True)


# %% [markdown]
# Copy the results to your jupyterhub environment, you can save a copy if you want (do not push it to gitlab)

# %%
# %spark -c False -o stop_cluster_loc_df -n -1

# %%
# # %%local
# stop_cluster_loc_df.to_csv("/home/jovyan/homework3/data/clusters.csv")

# %%
# %%local
import plotly.express as px
fig = px.scatter_mapbox(stop_cluster_loc_df, lat="stop_lat", lon="stop_lon", color="prediction", zoom=8, height=800, color_continuous_scale=px.colors.qualitative.Light24)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# %%
# %%local
import matplotlib.pyplot as plt
import seaborn as sns

cluster_counts = stop_cluster_loc_df['prediction'].value_counts()
cluster_counts

# %% [markdown]
# ---
# ## PART III: SBB Delay Model building (20 points)
#
# In the final segment of this assignment, your task is to tackle the prediction of SBB delays within the Lausanne region.
#
# To maintain simplicity, we've narrowed down the scope to building and validating a model capable of predicting delays exceeding 5 minutes. The model will classify delays as 0 if they're less than 5 minutes, and 1 otherwise.
#
# This problem offers ample room for creativity, allowing for multiple valid solutions. We provide a structured sequence of steps to guide you through the process, but beyond that, you'll navigate independently. By this stage, you should be adept in utilizing the Spark API, enabling you to explore the Spark documentation and gather all necessary information.
#
# Feel free to employ innovative approaches and leverage methods and data acquired in earlier sections of the assignment. This open-ended problem encourages exploration and experimentation.

# %%
import pyspark.sql.functions as F

# %% [markdown]
# ### a) Feature Engineering - 8/20
#
#
# Construct a feature vector for training and testing your model.
#
# Best practices include:
#
# * Data Source Selection and Exploration:
#   - Do not hesitate to reuse the data from Lausanne created in assignment 2. Query the data from your Hive database directly into Spark DataFrames.
#   - Explore the data to understand its structure, identifying relevant features and potential issues such as missing or null values.
#
# * Data Sanitization:
#   - Clean up null values and handle any inconsistencies or outliers in the data.
#   - Decide which fields to use: e.g. `bpuic` or `haltestellen_name` to identify the stops. Choose the one that provides better consistency and granularity for your model.
#
# * Historical Delay Computation:
#   - Utilize the SBB historical istdaten to compute historical delays, incorporating this information into your feature vector.
#   - Experiment with different ways to represent historical delays, such as aggregating delays over different time periods or considering average delays for specific routes or stations.
#
# * Incorporating Additional Data Sources:
#   - Experiment with integrating other relevant data sources, such as weather data and clustering data seen in the previous questions into your feature vector.
#   - Explore how these additional features contribute to the predictive power of your model and how they interact with the primary dataset.
#   - Note: the weather station location data is available under `/data/wunderground/csv/stations/`
#
# * Feature Vector Construction using Spark MLlib:
#   - Utilize [`Spark MLlib`](https://spark.apache.org/docs/latest/ml-features.html). methods to construct the feature vector for your model.
#   - Consider techniques such as feature scaling, transformation, and selection to enhance the predictive performance of your model.
#

# %%
spark.read.csv('/data/wunderground/csv/stations', header=True).printSchema()

# %%
spark.read.orc('/data/sbb/orc/istdaten').printSchema()

# %% [markdown]
# ### 1. Data Source Selection and Exploration

# %% [markdown]
# ### a. Loading the Data

# %% [markdown]
# #### i. From Lausanne created in assignment 2: `sbb_stops_lausanne_region_df`

# %%
spark.sql(f'CREATE DATABASE IF NOT EXISTS {username}')
spark.sql(f'SHOW TABLES IN {username}').toPandas()

# %%
sbb_stops_lausanne_region_df = spark.sql(f'SELECT * FROM {username}.sbb_stops_lausanne_region')
sbb_stops_lausanne_region_df.show(1)

# %% [markdown]
# > This data will be useful to selecg the stop_name ( buic or haltestellen_name)   that are comprises in lausanne region in the data set istdaten. It also provides the mappig of the stop_name in stop_id, this will be useful to 

# %% [markdown]
# #### ii. `/data/sbb/orc/istdaten`

# %%
istdaten_df = spark.read.orc('/data/sbb/orc/istdaten')
istdaten_df.printSchema()
istdaten_df.show(1, vertical=True)

# %% [markdown]
# > Pour le développement de notre modèle de prédiction des retards, certaines informations extraites du jeu de données `istdaten` sont particulièrement utiles. Nous allons limiter l'analyse aux données des années 2022 et 2023 pour plusieurs raisons : premièrement, cela réduit la charge computationnelle en diminuant le volume de données à traiter; deuxièmement, les données les plus récentes sont plus susceptibles de refléter les conditions actuelles, contrairement à des périodes exceptionnelles comme celle de la COVID-19 en 2020, qui pourrait introduire des biais dans notre modèle. Les colonnes spécifiques que nous retiendrons incluent :
# > - `betriebstag` : la date du voyage, nous permettant de cibler les données des années spécifiques.
# > - `betreiber_abk` : car l'ont veut simpelment etudier les retardzs de la sbb.
# > - `produkt_id` : le type de transport, car différents modes peuvent avoir des susceptibilités variées aux retards.
# > - `haltestellen_name` : le nom de l'arrêt, crucial pour lier ces données avec d'autres jeux de données géolocalisées.
# > - `ankunftszeit` et `an_prognose` : l'heure prévue et l'heure réelle d'arrivée, permettant de calculer directement les retards.

# %%
# Selection of the features of interest
selected_columns = [
    istdaten_df['betriebstag'].alias('date'),
    istdaten_df['fahrt_bezeichner'].alias('trip_id'),
    istdaten_df['betreiber_abk'].alias('operator'),
    istdaten_df['produkt_id'].alias('product'),
    istdaten_df['haltestellen_name'].alias('stop_name'),
    istdaten_df['ankunftszeit'].alias('actual_arrival'),
    istdaten_df['an_prognose'].alias('predicted_arrival_time'),
    istdaten_df['an_prognose_status'].alias('predicted_status')
]

# Creating a DataFrame with selected columns
istdaten_filtered_df = istdaten_df.select(*selected_columns)

# Filtering data for the years 2022 and 2023 and operator is SBB
istdaten_filtered_df = istdaten_filtered_df.filter(
    (istdaten_filtered_df['date'].substr(7, 4) == '2022') | (istdaten_filtered_df['date'].substr(7, 4) == '2023') &
    (istdaten_filtered_df['operator']  == 'SBB')
)

istdaten_filtered_df.show(5, vertical=True)

# %% [markdown]
# #### iii. `/data/wunderground/csv/stations`

# %%
wd_stations_df = spark.read.csv('/data/wunderground/csv/stations', header=True)
wd_stations_df.printSchema()
wd_stations_df.show(1, vertical=True)

# %% [markdown]
# #### iv. `/data/wunderground/json/history/`

# %%
df = spark.read.json('/data/wunderground/json/history/')
wd_history_df = df.select(df.valid_time_gmt, df.clds, df.day_ind, df.dewPt, df.feels_like, df.gust, 
                       df.heat_index, df.obs_name, df.precip_hrly, df.precip_total, df.pressure, 
                       df.rh, df.temp, df.uv_desc, df.uv_index, df.vis, df.wc, df.wdir, df.wdir_cardinal,
                       df.wspd, df.wx_phrase, df.site, 
                       F.year(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('year'),
                       F.month(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('month'),
                       F.dayofmonth(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('dayofmonth'),
                       F.hour(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('hour'),
                       F.minute(F.to_timestamp(F.from_unixtime(df.valid_time_gmt))).alias('minute'))
wd_history_df.printSchema()
wd_history_df.show(1, vertical=True)

# %% [markdown]
# #### v. Refining Clustering for Lausanne Stops

# %% [markdown]
# In Part II of our homework, we conducted clustering considering stops across all of Switzerland. However, to obtain more relevant information for our model, we plan to revisit the methods deployed in Part II to perform clustering with only the stops from Lausanne.
#
# The idea behind this approach is to address the following issue: during clustering in Part II, we identified an optimal number of 14 stop clusters across Switzerland. However, after filtering stop names to only consider Lausanne, we found that only 5 clusters were present in our data. Moreover, the frequency of presence did not match the size of the original cluster.
#
# The goal here is to redo clustering using the same methods as in Part II, but this time considering only Lausanne data. This will make these features relevant and potentially allow us to identify outliers, such as isolated stop names within a cluster.
#
# To achieve this, we will leverage the data available in the file `/data/share/stop_to_stop_lausanne_df.orc/`.
#

# %% [markdown]
# ##### 1. Load the data 

# %%
# Load the data for stops in Lausanne
stops_lausanne_df = spark.read.orc('/data/share/stop_to_stop_lausanne_df.orc/')

# Display the schema and the first row of the DataFrame
stops_lausanne_df.printSchema()
stops_lausanne_df.show(1, vertical=True)


# %% [markdown]
# ##### 2. Convert src_id and dst_id to numerical values

# %%
from pyspark.ml.feature import StringIndexer, IndexToString

# %%
# Initialize StringIndexer for source and destination IDs
string_indexer_src = StringIndexer(inputCol="src_id", outputCol="src_index")
string_indexer_dst = StringIndexer(inputCol="dst_id", outputCol="dst_index")

# Fit StringIndexer models
string_indexer_src_model = string_indexer_src.fit(stops_lausanne_df)
string_indexer_dst_model = string_indexer_dst.fit(stops_lausanne_df)

# Transform DataFrame to add indexed columns
stops_lausanne_df = string_indexer_src_model.transform(stops_lausanne_df)
stops_lausanne_df = string_indexer_dst_model.transform(stops_lausanne_df)

# Show the first row of the transformed DataFrame
stops_lausanne_df.show(1, vertical=True)

# %% [markdown]
# ##### 3. creation of the network_lausanne_df

# %%
# TODO: create network_df (hint: cast stop indices to integers)
network_lausanne_df = stops_lausanne_df.select(
    stops_lausanne_df.src_index.cast('int').alias("src"),          
    stops_lausanne_df.dst_index.cast('int').alias("dst"),          
    stops_lausanne_df.src_sequence.cast('int'),           
    stops_lausanne_df.dst_sequence.cast('int'), 
    stops_lausanne_df.trip_id
)

network_lausanne_df.show(1, vertical = True)

# %% [markdown]
# ##### 4. Weight for network_lausanne_df

# %%
# Calculate trip frequency between source and destination
trip_frequency_df = (network_lausanne_df
                     .groupBy("src", "dst")
                     .agg(F.count(network_lausanne_df.trip_id).alias("trip_frequency")))

# Calculate difference in stop sequence between source and destination
diff_dist_df = (network_lausanne_df
                .groupBy("src", "dst", "src_sequence", "dst_sequence")
                .agg(F.abs(network_lausanne_df.dst_sequence - network_lausanne_df.src_sequence).alias("stop_sequence_difference")))

# Join trip frequency and stop sequence difference DataFrames
weighted_network_df = trip_frequency_df.join(diff_dist_df, ["src", "dst"])

# Compute min and max values for normalization
min_max_values = weighted_network_df.select(
    F.min("trip_frequency").alias("min_frequency"),
    F.max("trip_frequency").alias("max_frequency"),
    F.min("stop_sequence_difference").alias("min_stop_diff"),
    F.max("stop_sequence_difference").alias("max_stop_diff")
).collect()[0]

# Normalize trip frequency and stop sequence difference
weighted_network_df = weighted_network_df.withColumn(
    "normalized_frequency",
    (weighted_network_df.trip_frequency - min_max_values.min_frequency) / (min_max_values.max_frequency - min_max_values.min_frequency)
).withColumn(
    "normalized_stop_diff",
    (weighted_network_df.stop_sequence_difference - min_max_values.min_stop_diff) / (min_max_values.max_stop_diff - min_max_values.min_stop_diff)
)

# Define a small epsilon value to prevent dividing by zero
epsilon = 1e-10

# Calculate weighted weight based on trip frequency and stop sequence difference
weighted_weight_df = weighted_network_df.withColumn('weighted_weight', 
                                    weighted_network_df.normalized_frequency / (weighted_network_df.normalized_stop_diff + epsilon))

# Select relevant columns for the network DataFrame
network_lausanne_df = weighted_weight_df.select(
    weighted_weight_df.src,
    weighted_weight_df.dst,
    (weighted_weight_df.weighted_weight).alias('weight')
)

# Display the network DataFrame
network_lausanne_df.show(1, vertical=True)
network_lausanne_df.printSchema()

# %%
network_lausanne_df.coalesce(1).write.parquet(f'/user/{username}/network_part_III', mode="overwrite")
network_lausanne_df = spark.read.parquet(f'/user/{username}/network_part_III')
network_lausanne_df.printSchema()
network_lausanne_df.show(1, vertical = True)

# %% [markdown]
# ##### 5. Cluster the stops network_lausanne_df

# %%
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import PowerIterationClustering

# %%
# Initialize lists to store k values and corresponding silhouette scores
k_values = []
silhouette_scores = []

# Initialize variable to store previous silhouette score
prev_silhouette = None

# Loop over k values from 8 to 20 with a step of 2
# for k in range(8, 21, 1):
for k in range(2, 10, 1):
    print(k)
    # Apply Clustering method on network_df
    pic = PowerIterationClustering(k=k, maxIter=50)
    assignments = pic.assignClusters(network_lausanne_df)
    assignments = network_df.join(assignments, on=[(network_df.src == assignments.id)]).drop('id')
    assignments = assignments.withColumnRenamed('weight', 'features')

    # Create a VectorAssembler
    vecAssembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")

    # Transform the DataFrame
    assignments = vecAssembler.transform(assignments)

    # Rename 'cluster' to 'prediction'
    assignments = assignments.withColumnRenamed('cluster', 'prediction')

    # Use the vectorized features for evaluation
    evaluator = ClusteringEvaluator(featuresCol="features_vec")
    silhouette = evaluator.evaluate(assignments)

    # Store k and silhouette score
    k_values.append(k)
    silhouette_scores.append(silhouette)
    print("k={}, silhouette_score={}".format(k,silhouette))

    # Check if silhouette score has improved
    if prev_silhouette is not None and silhouette <= prev_silhouette :
        break

    # Update previous silhouette score
    prev_silhouette = silhouette

# Plot silhouette scores as a function of k
plt.close('all')
plt.plot(k_values, silhouette_scores)
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score as a function of k')
plt.show()

# %%
# %matplot plt

# %%
#assignments.coalesce(1).write.parquet(f'/user/{username}/assignments_part_III', mode="overwrite")
assignments = spark.read.parquet(f'/user/{username}/assignments_part_III')
assignments.printSchema()
assignments.show(1, vertical = True)

# %%
prediction_distribution_df = (assignments.groupBy('prediction')
                                   .count()
                                   .orderBy('count', ascending=False)
                                   .toPandas())

# Afficher la répartition des prédictions
print(prediction_distribution_df)

# Afficher un histogramme
plt.close('all')
plt.figure(figsize=(10, 6))
plt.bar(prediction_distribution_df['prediction'], prediction_distribution_df['count'], color='skyblue')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.title('Prediction Distribution')
plt.show()

# %%
# Initiate IndexToString for converting stop indices to stop names for both src and dst columns
index_to_string_src = IndexToString(inputCol="src", outputCol="stop_id_src").setLabels(string_indexer_src_model.labels)

assignments_with_names = index_to_string_src.transform(assignments).select('stop_id_src', 'prediction')

assignments_with_names.printSchema()
assignments_with_names.show(1, vertical = True)

# %%
#assignments_with_names.coalesce(1).write.parquet(f'/user/{username}/assignments_names_part_III', mode="overwrite")
assignments_with_names = spark.read.parquet(f'/user/{username}/assignments_names_part_III')
assignments_with_names.printSchema()
assignments_with_names.show(1, vertical = True)

# %%
stops_df = spark.read.orc('/data/sbb/orc/timetables/stops')

stop_cluster_loc_df = assignments_with_names.join(stops_df, assignments_with_names.stop_id_src == stops_df.stop_id).select('stop_id', 'stop_lat', 'stop_lon', 'prediction')
stop_cluster_loc_df.printSchema()
stop_cluster_loc_df.show(1, vertical = True)

# %%
stop_cluster_loc_df.coalesce(1).write.parquet(f'/user/{username}/stop_cluster_loc_part_III', mode="overwrite")
stop_cluster_loc_df = spark.read.parquet(f'/user/{username}/stop_cluster_loc_part_III')
stop_cluster_loc_df.printSchema()
stop_cluster_loc_df.show(1, vertical = True)

# %%
# %spark -c False -o stop_cluster_loc_df -n -1

# %%
# %%local
import plotly.express as px
fig = px.scatter_mapbox(stop_cluster_loc_df, lat="stop_lat", lon="stop_lon", color="prediction", zoom=8, height=800, color_continuous_scale=px.colors.qualitative.Light24)
fig.update_layout(mapbox_style="open-street-map")
fig.show()

# %% [markdown]
# ### b. Integration of the data 

# %% [markdown]
# #### Merging `sbb_stops_lausanne_region_df` and  `istdaten_filtered_df`

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# > We know from Assignment 2 that the DataFrame `sbb_stops_lausanne_region_df` contains only the `stop_id` and `stop_name` for the Lausanne region. To make our dataset relevant to our regional study, we will first filter the `istdaten` dataset to include only those records where the stop names match those found in the Lausanne-specific DataFrame. This will help ensure that our model predictions are specifically tailored to the Lausanne area and not influenced by data from outside this region.
#

# %%
# Joining DataFrames on 'stop_name' and dropping the column 'stop_name'
istdaten_filtered_lausanne_df = istdaten_filtered_df.join(
    sbb_stops_lausanne_region_df, 
    istdaten_filtered_df.stop_name == sbb_stops_lausanne_region_df.stop_name,
    "inner"
)

# Dropping the 'stop_name' column
istdaten_filtered_lausanne_df = istdaten_filtered_lausanne_df.drop(sbb_stops_lausanne_region_df.stop_name)

# Displaying the first row vertically
istdaten_filtered_lausanne_df.show(1, vertical=True)


# %% [markdown]
# #### Merging resulting `istdaten_filtered_df` and  `stop_cluster_loc_df`

# %% [markdown]
# > add a comment 

# %%
stop_cluster_loc_df = spark.read.parquet(f'/user/{username}/stop_cluster_loc')
stop_cluster_loc_df.printSchema()
stop_cluster_loc_df.show(1, vertical = True)

# %%
# Joining DataFrames on 'stop_id' and dropping the column 'stop_lat' and 'stop_lon'
istdaten_with_cluster_df = istdaten_filtered_lausanne_df.join(
    stop_cluster_loc_df.select('stop_id', stop_cluster_loc_df.prediction.alias('cluster')),
    ['stop_id'],
    "inner"
)

istdaten_with_cluster_df.show(1, vertical=True)


# %%
istdaten_with_cluster_df.coalesce(1).write.parquet(f'/user/{username}/istdaten_with_cluster', mode="overwrite")
istdaten_with_cluster_df = spark.read.parquet(f'/user/{username}/istdaten_with_cluster')
istdaten_with_cluster_df.printSchema()
istdaten_with_cluster_df.show(1, vertical = True)


# %% [markdown]
# ### c. Data Sanitization

# %% [markdown]
# > Avant de creer la data pour ntoer mdoel nous devons clean up null valuse an dhandle any incosistancies or outliers in the data. 

# %% [markdown]
# #### i. Clean up null values
#
# 1. **Non-empty and Non-null `operator`**: Ensures that only entries with valid product types are included.
# 2. **Non-null `actual_arrival` (Arrival Time)**: Guarantees that the dataset includes only records where the actual arrival time is known.
# 3. **`predicted_status` in ('REAL', 'GESCHAETZT')**: Filters the data to include only records where the predicted arrival status is either "REAL" (actual) or "GESCHAETZT" (estimated), ensuring the predictions are based on realistic or estimated data rather than missing or undefined statuses.
# 4. **Non-empty and Non-null `stop_name`**: Ensures that records have valid stop names, which is critical for joining with other location-based datasets.

# %%
data_df = istdaten_with_cluster_df.filter(
    (~F.isnull(istdaten_with_cluster_df.operator)) &
    (~F.isnull(istdaten_with_cluster_df.product)) &
    (istdaten_with_cluster_df.product != "") &
    (~F.isnull(istdaten_with_cluster_df.actual_arrival)) &
    (istdaten_with_cluster_df.predicted_status.isin('REAL', 'GESCHAETZT')) &
    (istdaten_with_cluster_df.stop_name != "") &
    (~F.isnull(istdaten_with_cluster_df.stop_name))
)

data_df = data_df.dropDuplicates()

data_df.printSchema()
data_df.show(5, vertical = True)

# %% [markdown]
# #### ii. Create column year, month and day

# %% [markdown]
# > checker si date est consistent avec prected arrival time  

# %%
data_df = data_df.withColumn('date', F.to_date(data_df.date, 'dd.MM.yyyy'))
data_df = data_df.withColumn('predicted_arrival_time', F.to_timestamp(data_df.predicted_arrival_time, 'dd.MM.yyyy HH:mm:ss'))
data_df = data_df.withColumn('actual_arrival', F.to_timestamp(data_df.actual_arrival, 'dd.MM.yyyy HH:mm'))


data_df = data_df.select(
    data_df.stop_name, 
    data_df.stop_id,
    data_df.date, 
    F.year(data_df.date).alias('year'), 
    F.month(data_df.date).alias('month'), 
    F.dayofmonth(data_df.date).alias('day'), 
    F.hour(data_df.predicted_arrival_time).alias('predicted_hour'),
    data_df.predicted_arrival_time, 
    data_df.actual_arrival, 
    data_df.trip_id, 
    data_df.product, 
    data_df.cluster, 
    data_df.stop_lat, 
    data_df.stop_lon
)

data_df.printSchema()
data_df.show(1, vertical=True)

# %%
#data_df.coalesce(1).write.parquet(f'/user/{username}/data_before_label', mode="overwrite")
data_df = spark.read.parquet(f'/user/{username}/data_before_label')
data_df.printSchema()
data_df.show(1, vertical = True)


# %% [markdown]
# #### iii. Create label for classifer 

# %%
def calculate_label(predicted_arrival, actual_arrival):
    predicted_str = F.date_format(predicted_arrival, 'yyyy-MM-dd HH:mm:ss')
    actual_str = F.date_format(actual_arrival, 'yyyy-MM-dd HH:mm:ss')
    time_difference = F.abs(F.unix_timestamp(actual_str) - F.unix_timestamp(predicted_str)) / 60  
    return F.when(time_difference <= 5, 0).otherwise(1)

data_df = data_df.withColumn('label', calculate_label(data_df.predicted_arrival_time, data_df.actual_arrival))

data_df.printSchema()
data_df.show(1, vertical=True)

# %%
data_df.coalesce(1).write.parquet(f'/user/{username}/data_label', mode="overwrite")
data_df = spark.read.parquet(f'/user/{username}/data_label')
data_df.printSchema()
data_df.show(1, vertical=True)

# %% [markdown]
# ### d. Historical Delay Computation 

# %% [markdown]
# #### Monthly and Product Average Delay Analysis

# %%
import matplotlib.pyplot as plt

# Group the data by month and product, and calculate the average delay
monthly_product_avg_delay = data_df.groupBy('month', 'product').agg(F.avg('label').alias('avg_delay_month_product'))

# Sort the results by month and product
monthly_product_avg_delay = monthly_product_avg_delay.orderBy('month', 'product')

# Collect the data into a Pandas DataFrame for displaying the histogram
monthly_product_avg_delay_pandas = monthly_product_avg_delay.toPandas()

# Get unique months and products
unique_months = monthly_product_avg_delay_pandas['month'].unique()
unique_products = monthly_product_avg_delay_pandas['product'].unique()

# Define bar width
bar_width = 0.35

# Set positions for the bars
bar_positions = range(len(unique_months))

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the bars for each product
for i, product in enumerate(unique_products):
    product_data = monthly_product_avg_delay_pandas[monthly_product_avg_delay_pandas['product'] == product]
    ax.bar([pos + i * bar_width for pos in bar_positions], 
           product_data['avg_delay_month_product'], 
           bar_width, label=product)

# Set x-axis labels to be the unique months
ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
ax.set_xticklabels(unique_months.astype(str))

# Set labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Average Delays')
ax.set_title('Average Delays per Month and Product')

# Add legend
ax.legend()

# Show plot
plt.show()

# %matplot plt

# %% [markdown]
# > We observe two things: a difference between the products, buses are much more likely to be delayed, and between the months of the year. For trains, for example ('ZUG'), the month of November seems to be more impacted than the month of August, for instance.

# %%
# Join monthly_avg_delay with data_df on the month column
data_df = data_df.join(monthly_product_avg_delay, on=['month', 'product'], how='left')
data_df.printSchema()
data_df.show(1, vertical=True)

# %% [markdown]
# #### Average Delay per hour 

# %%
avg_delay_per_hour = data_df.groupBy('predicted_hour').agg(F.avg('label').alias('avg_delay_per_hour'))
avg_delay_per_hour.show()

# %%
import matplotlib.pyplot as plt

# Collect the data into a Pandas DataFrame for displaying the histogram
avg_delay_per_hour_pandas = avg_delay_per_hour.toPandas()

# Display a histogram of average delays per predicted hour
plt.figure(figsize=(12, 8))
plt.bar(avg_delay_per_hour_pandas['predicted_hour'], avg_delay_per_hour_pandas['avg_delay_per_hour'])
plt.xlabel('Predicted Hour')
plt.ylabel('Average Delay')
plt.title('Average Delay per Predicted Hour')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %matplot plt

# %%
data_df = data_df.join(avg_delay_per_hour, on='predicted_hour', how='left')
data_df.printSchema()
data_df.show(1, vertical=True)

# %% [markdown]
# #### Average Delay per trip_id

# %%
avg_delay_per_trip_id = data_df.groupBy('trip_id').agg(F.avg('label').alias('avg_delay_per_trip_id'))
avg_delay_per_trip_id.show()

# %%
data_df = data_df.join(avg_delay_per_trip_id, on='trip_id', how='left')
data_df.printSchema()
data_df.show(1, vertical=True)

# %%
#data_df.coalesce(1).write.parquet(f'/user/{username}/data_delay', mode="overwrite")
data_df = spark.read.parquet(f'/user/{username}/data_delay')
data_df.printSchema()
data_df.show(1, vertical=True)

# %% [markdown]
# ### e. Incorportating Additionnal Sources

# %%
wd_stations_df = spark.read.csv('/data/wunderground/csv/stations', header=True)
wd_stations_df.printSchema()
wd_stations_df.show(1, vertical=True)

# %%
avg_monthly_precip_df = spark.read.parquet(f'/user/{username}/avg_monthly_precip')

avg_monthly_precip_df.printSchema()
avg_monthly_precip_df.show(1, vertical = True)

# %%
window_temp_hour = spark.read.parquet(f'/user/{username}/window_temp_site')
window_temp_hour.printSchema()
window_temp_hour.show(1, vertical = True)

# %% [markdown]
# #####  Merging additionnal data together 

# %%
# adding longitude and latitude from wd_stations_df 
avg_monthly_precip_df = avg_monthly_precip_df.join(wd_stations_df.select('site', 'lat_wgs84', 'lon_wgs84'), on='site', how='left')
avg_monthly_precip_df.printSchema()
avg_monthly_precip_df.show(1, vertical = True)

# %%
# Merge window_temp_hour and avg_monthly_precip_df on site and month 
additional_data = window_temp_hour.join(
    avg_monthly_precip_df, 
    on=['site', 'month']
)
additional_data.show(1, vertical=True)

# %% [markdown]
# > How to merge since lon and lat are certainly not the same 

# %%
data_df.printSchema()

# %% [markdown]
# ### f. Feature Vector Construction using Spark MLib 

# %%
# selection of the features 
data_features = data_df.select(
            data_df.stop_name, 
            data_df.product, 
            data_df.cluster, 
            data_df.avg_delay_month_product, 
            data_df.avg_delay_per_hour, 
            data_df.avg_delay_per_trip_id, 
            data_df.label
    
)
data_features.printSchema()
data_features.show(1, vertical = True)

# %% [markdown]
# #### Preprocess the data 

# %%
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# %%
# Transform categorical variable into numerical
columns_to_index = ['stop_name', 'product']

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data_features) for column in columns_to_index]

# Create a pipeline with StringIndexer stages
pipeline = Pipeline(stages=indexers)

# Fit and transform the data
data_features = pipeline.fit(data_features).transform(data_features)

# Drop the original columns
data_features = data_features.drop(*columns_to_index)

data_features.printSchema()
data_features.show(1, vertical=True)

# %%
from pyspark.ml.feature import VectorAssembler, StandardScaler

# %%
# Scale continuous features 

# Assemble continuous features for scaling
continuous_features_assembler = VectorAssembler(
    inputCols=["avg_delay_month_product", "avg_delay_per_hour", "avg_delay_per_trip_id"],
    outputCol="continuous_features"
)
data_features = continuous_features_assembler.transform(data_features)

#  Scale continuous features
scaler = StandardScaler(inputCol="continuous_features", outputCol="scaled_continuous_features")
data_features = scaler.fit(data_features).transform(data_features)

# %%
# Assemble all features into a single vector
feature_assembler = VectorAssembler(
    inputCols=["stop_name_index", "product_index", "cluster", "scaled_continuous_features"],
    outputCol="features"
)

data_features = feature_assembler.transform(data_features)
data_features.printSchema()
data_features.show(1, vertical=True)

# %%
model_data = data_features.select(
    data_features.features, 
    data_features.label
)
model_data.show(1, vertical=True)

# %%
#model_data.coalesce(1).write.parquet(f'/user/{username}/model_data', mode="overwrite")
model_data = spark.read.parquet(f'/user/{username}/model_data')
model_data.printSchema()
model_data.show(1, vertical = True)

# %% [markdown]
# ### b) Model building - 6/20
#
# Utilizing the features generated in section III.a), your objective is to construct a model capable of predicting delays within the Lausanne region. You have the option to reuse the stops identified in assignment 2 or utilize the stops made available to you under `/data/sbb/share`.
#
# To accomplish this task effectively:
#
# * Feature Integration:
#         - Incorporate the features created in section III.a) into your modeling pipeline.
#
# * Model Selection and Training:
#         - Explore various machine learning algorithms available in Spark MLlib to identify the most suitable model for predicting delays.
#         - Train the selected model using the feature vectors constructed from the provided data.

# %%
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=1234)

# Print the count of rows in each dataset to verify the split
print("Training Dataset Count: " + str(train_data.count()))
print("Testing Dataset Count: " + str(test_data.count()))

# %%
from pyspark.ml.classification import LogisticRegression

# Initialize the Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Train the model on the training data
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# %% [markdown]
# ### c) Model evaluation - 6/20
#
# * Evaluate the performance of your model
#     * Usie appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
#     * Utilize techniques such as cross-validation to ensure robustness and generalizability of your model.
#
# * Interpretation and Iteration:
#     * Interpret the results of your model to gain insights into the factors influencing delays within the Lausanne region.
#     * Iterate III.a)on your model by fine-tuning hyperparameters, exploring additional feature engineering techniques, or experimenting with different algorithms to improve predictive performance.
#

# %%
# from https://medium.com/@demrahayan/evaluating-binary-classification-models-with-pyspark-2afc5ac7937f
from pyspark.sql.functions import col

def get_evaluations(predictions):
    tp = predictions.filter((col('label') == 1) & (col('prediction') == 1)).count()
    tn = predictions.filter((col('label') == 0) & (col('prediction') == 0)).count()
    fp = predictions.filter((col('label') == 0) & (col('prediction') == 1)).count()
    fn = predictions.filter((col('label') == 1) & (col('prediction') == 0)).count()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy}")

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  
    print(f"Precision: {precision}")

    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0  
    print(f"Recall: {recall}")

    f1_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0  
    print(f"F1 measure: {f1_measure}")


# %%
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# crossvalidation
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)  # Number of folds for cross-validation

# Train the model using cross-validation
cv_model = crossval.fit(train_data)

# Make predictions on the test data
predictions = cv_model.transform(test_data)

# %%
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# %%
get_evaluations(predictions)

# %% [markdown]
# It seems that doing cross validation on different regularization parameters only improves our model in a minor way. The main the model has is that it does not pick up on a lot of cases where there are delays. This is probably because the training data is not balanced (most of the time they are on time). We should re-fit our model with weighted labels and improve the recall of the model.

# %%
# see label imbalance
class_distribution = predictions.groupBy("label").count().orderBy("label")
class_distribution.show()

# %%
from pyspark.ml.evaluation import Evaluator

class F1Eval(Evaluator):
    def __init__(self):
        super(F1Eval, self).__init__()
    
    def _evaluate(self, dataset):
        tp = dataset.filter((dataset['label'] == 1) & (dataset['prediction'] == 1)).count()
        tn = dataset.filter((dataset['label'] == 0) & (dataset['prediction'] == 0)).count()
        fp = dataset.filter((dataset['label'] == 0) & (dataset['prediction'] == 1)).count()
        fn = dataset.filter((dataset['label'] == 1) & (dataset['prediction'] == 0)).count()

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0  
        
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0  

        f1_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0  
        return f1_measure
    
    def isLargerBetter(self):
        return True  



# %%
from pyspark.sql.functions import when

class_weights = {
    0: 1.0,  
    1: 20.0  
}

# adding weights based on class labels
weighted_data = train_data.withColumn("weight", when(predictions["label"] == 0, class_weights[0]).otherwise(class_weights[1]))

# training model
lr = LogisticRegression(featuresCol='features', labelCol='label', weightCol='weight')  # Specify the weight column
evaluator = F1Eval()

# parameter grid for cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# training model using cross-validation
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)  # Number of folds for cross-validation

cv_model = crossval.fit(weighted_data)

predictions = cv_model.transform(test_data)

# %%
get_evaluations(predictions)

# %% [markdown]
# Re-weighting the labels to balance the classes and doing cross-validation on the regularization parameters improved our model by a factor of two when looking at the F1-Score. Still the model is far from perfect and would have to be improved in order to be useful in an operational level. Improvements could be done by finding other features to predict delays and/or trying different model architectures

# %% [markdown]
# You can copy the results with `%%spark -o ...` as shown before and visualize them in your notebook

# %% [markdown]
# # That's all, folks!
#
# Be nice to other, do not forget to close your spark session.

# %%
spark.stop()
