# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Homework 2 - Data Wrangling with Hadoop
# ---
#
# The goal of this assignment is to put into action the data wrangling techniques from the exercises of module 2. We highly suggest you to finish these two exercises first and then start the homework. In this homework, we are going to reuse the __sbb__ datasets. 
#
# ## Hand-in Instructions
# - __Due: 09.04.2024 23:59 CET__
# - `git push` your final verion to your group's git repository before the due date
# - Verify that `environment.yml` and `requirements.txt` are updated if you added new packages, and notebook is functional
# - Do not commit notebooks with the results, do a `Restart Kernel and Clear All Outputs...`
# - Add necessary comments and discussion to make your queries human readable
#
# ## Useful references
#
# * Hive queries: <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Select>
# * Hive functions: <https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF>
# * [ESRI GIS API](https://github.com/Esri/spatial-framework-for-hadoop/wiki/UDF-Documentation)
# * [Enclosed or Unenclosed](https://github.com/Esri/spatial-framework-for-hadoop/wiki/JSON-Formats)

# ---
# ⚠️ **Note**: all the data used in this homework is described in the [FINAL-PREVIEW](../final-preview.md) document, which can be found in this repository. The document describes the final project due for the end of this semester.
#
# For this notebook you are free to use the following tables, which can all be found under the `com490` database.
# - You can list the tables with the command `SHOW TABLES IN com490`.
# - You can see the details of each table with the command `DESCRIBE EXTENDED com490.{table_name}`.
#
# * com490.sbb_orc_calendar
# * com490.sbb_orc_stop_times
# * com490.sbb_orc_stops
# * com490.sbb_orc_trips
# * com490.sbb_orc_istdaten
# * com490.geo_shapes
#
# They are all part of the public transport timetable that was published on the week of 10.1.2024.
#
# ---
# For your convenience we also define useful python variables:
#
# * default_db=`com490`
#     * The Hive database shared by the class, do not drop or modify the content of this database.
# * hadoop_fs=`hdfs://iccluster067.iccluster.epfl.ch:8020`
#     * The HDFS server, in case you need it for hdfs commands with hive, pandas or pyarrow
# * username:
#     * Your user id (EPFL gaspar id), use it as your default database name.

# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Fair cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets. Do not hesitate to partion your tables, and LIMIT the output of your Hive queries to a few rows to begin with.
#     <br><br>
#     You may lose your hive session if you remain idle for too long or if you interrupt a query. If that happens you will not lose your tables, but you may need to reconnect to Hive and recreate the temporary UDF.
#     <br><br>
#     <b>Try to use as much HiveQL as possible and avoid using pandas operations.</b>
# </div>

# ---

# +
import os
from pyhive import hive
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

default_db = 'com490'
hive_server = os.environ.get('HIVE_SERVER','iccluster080.iccluster.epfl.ch:10000')
hadoop_fs = os.environ.get('HADOOP_DEFAULT_FS','hdfs://iccluster067.iccluster.epfl.ch:8020')
username  = os.environ.get('USER', 'anonym')
(hive_host, hive_port) = hive_server.split(':')

conn = hive.connect(
    host=hive_host,
    port=hive_port,
    username=username
)

# create cursor
cur = conn.cursor()

print(f"hadoop hdfs URL is {hadoop_fs}")
print(f"your username is {username}")
print(f"you are connected to {hive_host}:{hive_port}")
# -


# ---
# ### Part I. 10 Points

# ### a) Type of transport - 10/10
#
# In earlier exercises, you have already explored the stop distribution of different types of transport. Now, let's do the same for the whole of 2023 and visualize it in a bar graph.
#
# - Query `com490.sbb_orc_istdaten` to get the total number of stops for different types of transport in each month, and order it by time and type of transport.
# |month_year|ttype|stops|
# |---|---|---|
# |...|...|...|
# - Create a facet bar chart of monthly counts, partitioned by the type of transportation. 
# - If applicable, document any patterns or abnormalities you can find.
#
# __Note__: 
# - In general, one entry in the sbb istdaten table means one stop.
# - You might need to filter out the rows where:
#     - `BETRIEBSTAG` is not in the format of `__.__.____`
#     - `PRODUKT_ID` is NULL or empty
# - We recommend the facet _bar_ plot with plotly: https://plotly.com/python/facet-plots/ the monthly count of stops per transport mode as shown below:
#
# ```
# fig = px.bar(
#     df_ttype, x='month_year', y='stops', color='ttype',
#     facet_col='ttype', facet_col_wrap=3, 
#     facet_col_spacing=0.05, facet_row_spacing=0.2,
#     labels={'month_year':'Month', 'stops':'#stops', 'ttype':'Type'},
#     title='Monthly count of stops'
# )
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_layout(showlegend=False)
# fig.show()
# ```
#
#
# <img src="../notebooks/1a-example.png" alt="1a-example.png" width="400"/>

import pandas as pd
import plotly.express as px

# +
# %%time
query = f"""
    SELECT 
        SUBSTRING(BETRIEBSTAG, 4, 7) AS month_year,
        LOWER(PRODUKT_ID) AS ttype, 
        COUNT(*) AS stops
    FROM 
        {default_db}.sbb_orc_istdaten
    WHERE 
        PRODUKT_ID IS NOT NULL
        AND PRODUKT_ID <> ''
        AND year = 2023
        AND BETRIEBSTAG LIKE '__.__.2023'
    GROUP BY 
        SUBSTRING(BETRIEBSTAG, 4, 7), LOWER(PRODUKT_ID)
"""

df_ttype = pd.read_sql(query, conn)
display(df_ttype)
# -

fig = px.bar(
    df_ttype, x='month_year', y='stops', color='ttype',
    facet_col='ttype', facet_col_wrap=3, 
    facet_col_spacing=0.05, facet_row_spacing=0.2,
    labels={'month_year':'Month', 'stops':'#stops', 'ttype':'Type'},
    title='Monthly count of stops'
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_layout(showlegend=False)
fig.show()

#  It is unclear what the CS and WM-BUS types are. These transportation types are available in may 2023 and december 2023. The number of stops is very low compared to the other transportation types. These are probably temporary transport types put in place for specific events, maintenance, or alternative routes during construction work.

# ---
# ### Part II. 50 Points

# In this second Part, we will leverage Hive to model the public transport infrastructure within the Lausanne region.
#
# Our objective is to establish a comprehensive data representation of the public transport network, laying the groundwork for our final project. While we encourage the adoption of a data structure tailored to the specific requirements of your final project implementation, the steps outlined here provide a valuable foundation.
#
# In this part you will make good use of DQL statements of nested SELECT, GROUP BY, JOIN, IN, DISTINCT etc.

# ### a) Enable support for ESRI UDF - 2/50

# Use what you have learned in the exercises to enable ESRI User Defined Functions. At a minimum enable the UDF and SerDe functions required in order to complete this exercise.

# We first add the jars that are required for the ESRI UDF functions.
cur.execute(
    f"""
ADD JARS
    {hadoop_fs}/data/jars/esri-geometry-api-2.2.4.jar
    {hadoop_fs}/data/jars/spatial-sdk-hive-2.2.0.jar
    {hadoop_fs}/data/jars/spatial-sdk-json-2.2.0.jar
"""
)
cur.execute("LIST JARS")
print(cur.fetchall())

# We then create the temporary functions required for the ESRI UDF functions.
cur.execute("CREATE TEMPORARY FUNCTION ST_Point AS 'com.esri.hadoop.hive.ST_Point'")
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_Distance AS 'com.esri.hadoop.hive.ST_Distance'"
)
cur.execute("CREATE TEMPORARY FUNCTION ST_SetSRID AS 'com.esri.hadoop.hive.ST_SetSRID'")
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_GeodesicLengthWGS84 AS 'com.esri.hadoop.hive.ST_GeodesicLengthWGS84'"
)
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_LineString AS 'com.esri.hadoop.hive.ST_LineString'"
)
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_AsBinary AS 'com.esri.hadoop.hive.ST_AsBinary'"
)
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_PointFromWKB AS 'com.esri.hadoop.hive.ST_PointFromWKB'"
)
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_GeomFromWKB AS 'com.esri.hadoop.hive.ST_GeomFromWKB'"
)
cur.execute(
    "CREATE TEMPORARY FUNCTION ST_Contains AS 'com.esri.hadoop.hive.ST_Contains'"
)

cur.execute("SHOW FUNCTIONS")
for result in cur.fetchall():
    if result[0].startswith("st_"):
        print(result)

# ## b) Declare your database and your tables  - 2/50

# Establish a Hive-managed database under your designated database, ({username}), along with Hive tables essential for modeling your infrastructure. Should you opt for Hive management, omit the location and the external keyword when creating the database and the tables.
#
# At a minimum, you need two tables as follow. Use the provided information to decide the best types for the fields.
#
# * _{username}.sbb_stops_lausanne_region_ (subset of _com490.sbb_orc_stops_)
#     * `stop_id`
#     * `stop_name`
#     * `stop_lat`
#     * `stop_lon`
#
# * _{username}.sbb_stop_to_stop_lausanne_region_
#     * `stop_id_a`: an sbb_orc_stops.stop_id
#     * `stop_id_b`: an sbb_orc_stops.stop_id
#     * `distance`: straight line distance in meters from stop_id_a to stop_id_b
#
# * _{username}.sbb_stop_times_lausanne_region_ (subset of _com490.sbb_orc_stop_times_)
#     * `trip_id`
#     * `stop_id`
#     * `departure_time`
#     * `arrival_time`
#  
# Note: the time units of _{username}.sbb_stop_times_lausanne_region.{departure_time,arrival_time}_ do not need to be the same as _com490.sbb_orc_stop_times_. Feel free to take advantage of Hive's [Date UDF](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF#LanguageManualUDF-DateFunctions) to convert them into units that are more suitable for you. You can also break them into separate fields.

# #### 1. Create a Database

# +
query = f'CREATE DATABASE IF NOT EXISTS {username}'
cur.execute(query)

query = f"USE {username}"
cur.execute(query)

query = f"SHOW TABLES"
cur.execute(query)
cur.fetchall()
# -

# #### 2. Create sbb_stops_lausanne_region

# ##### a. Create table in my database

# Creation of the table sbb_stops_lausanne_region
query = f"""
    CREATE TABLE IF NOT EXISTS {username}.sbb_stops_lausanne_region (
        stop_id STRING,
        stop_name STRING,
        stop_lat DOUBLE,
        stop_lon DOUBLE
    )
    STORED AS ORC
"""
cur.execute(query)

# ##### b. Insert data into sbb_stops_lausanne_region

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_stops LIMIT 5"
pd.read_sql(query, conn)

# %%time
# Insert data into sbb_stops_lausanne_region table
query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stops_lausanne_region 
    SELECT 
        stop_id, 
        stop_name,
        stop_lat,
        stop_lon 
    FROM
        {default_db}.sbb_orc_stops
    WHERE
        stop_lat BETWEEN 46.486 AND 46.613
        AND stop_lon BETWEEN 6.524 AND 6.775
"""
cur.execute(query)

# check the content of the table 
query = f"SELECT * FROM {username}.sbb_stops_lausanne_region  LIMIT 5"
pd.read_sql(query, conn)

# #### 3. Create sbb_stop_to_stop_lausanne_region

# ##### a. Create table in my database

# +
# Creation of the table sbb_stop_to_stop_lausanne_region
query = f"""
    CREATE TABLE IF NOT EXISTS {username}.sbb_stop_to_stop_lausanne_region (
        stop_id_a STRING,
        stop_id_b STRING,
        distance DOUBLE
    )
    STORED AS ORC
"""

cur.execute(query)
# -

# ##### b. Insert data into sbb_stop_to_stop_lausanne_region

# %%time
# Insert data into sbb_stop_to_stop_lausanne_region
query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stop_to_stop_lausanne_region 
    SELECT 
        a.stop_id AS stop_id_a,
        b.stop_id AS stop_id_b,
        ST_GeodesicLengthWGS84(
                ST_SetSRID(ST_LineString(a.stop_lon,a.stop_lat,b.stop_lon,b.stop_lat), 4326)) AS distance
    FROM 
        {username}.sbb_stops_lausanne_region AS a 
    CROSS JOIN 
        {username}.sbb_stops_lausanne_region AS b
"""
cur.execute(query)

# check the content of the table 
query = f"SELECT * FROM {username}.sbb_stop_to_stop_lausanne_region LIMIT 5"
pd.read_sql(query, conn)

# #### 4. Create sbb_stop_times_lausanne_region

# ##### a. Create table in my database

query = f"""
    CREATE TABLE IF NOT EXISTS {username}.sbb_stop_times_lausanne_region(
        trip_id STRING, 
        stop_id STRING,
        departure_time STRING, 
        arrival_time STRING
    )
    STORED AS ORC"""
cur.execute(query) 


# ##### b. Insert data into sbb_stop_times_lausanne_region

# check if sbb_orc_stop_times is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_stop_times LIMIT 5"
pd.read_sql(query, conn)

# %%time
# Insert data into sbb_stop_times_lausanne_region table
query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stop_times_lausanne_region
    SELECT 
        trip_id, 
        stop_id, 
        departure_time, 
        arrival_time
    FROM 
        {default_db}.sbb_orc_stop_times
    """
cur.execute(query)


# check the content of the table 
query = f"SELECT * FROM {username}.sbb_stop_times_lausanne_region  LIMIT 5"
pd.read_sql(query, conn)

# ### c) Find the stops in Lausanne region - 5/50

# * Find all the stops from _com490.sbb_orc_stops_ that are contained in the _com490.geo_shapes.geometry_ where _com490.geo_shapes.objectid_=1.
#     * The shape is from metabolismofcities, you can visualize it [here](https://data.metabolismofcities.org/library/751445/) 
# * Save the results in _{username}.sbb_stops_lausanne_region_
# * Validation: you should find around $600\pm 20$ stops.
# * Note: you must complete a) and b) first.

# ##### a. Check if {default_db}.sbb_orc_stops and {default_db}.geo_shape not empty

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_stops LIMIT 5"
pd.read_sql(query, conn)

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.geo_shapes LIMIT 5"
pd.read_sql(query, conn)

# ##### b. Find the stop where in Lausanne region and contain in geoshapes.geometry

query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stops_lausanne_region
    SELECT
        a.stop_id,
        a.stop_name,
        a.stop_lat,
        a.stop_lon
    FROM 
        {default_db}.sbb_orc_stops a 
    JOIN 
        {default_db}.geo_shapes b
    WHERE
        b.objectid=1 
        AND ST_Contains(b.geometry, ST_Point(a.stop_lon,a.stop_lat))
"""
cur.execute(query)

# check the result in sbb_stops_lausanne_region 
query = f"SELECT * FROM {username}.sbb_stops_lausanne_region"
df_stops_lausanne  = pd.read_sql(query, conn)


df_stops_lausanne.head()

# ##### c. Validation 

# %%time
query = f"""
    SELECT COUNT(*) FROM {username}.sbb_stops_lausanne_region
"""
pd.read_sql(query, conn)

# ### d) Find stops with real time data in Lausanne region - 4/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the stops for which real time data is reported in the _com490.sbb_orc_istdaten_ table for the month of January 2024.
# * Use plotly to display all the stop locations in Lausanne region on a map, using a different color to highlight the stops for which istdaten data is available
# * Validation: you should find around $580\pm 20$ stops.
# * Note: you must complete c) first.
# * Hints:
#     - It is recommended to first generate a list of _distinct_ stop IDs extracted from istdaten data for January 2024. This can be achieved through either a nested query or by creating an intermediate table.
#     - Be aware that there may be inconsistencies among public transport operators when reporting stop IDs in istdaten. 

# ##### a. Check if {default_db}.sbb_orc_stops not empty

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_istdaten LIMIT 5"
pd.read_sql(query, conn)

# ##### b. Create an intermediate table which only contain stop name and stop id distinct which are real and dated from January 2024

# +
# Create intermediate view
query = f"""
    CREATE VIEW {username}.intermediate
    AS 
    SELECT DISTINCT 
        a.haltestellen_name as stop_name,
        a.betreiber_id as stop_id
    FROM 
        {default_db}.sbb_orc_istdaten AS a
    WHERE 
        a.year = 2024 AND 
        a.month = 1 AND
        a.haltestellen_name <> '' AND 
        a.haltestellen_name IS NOT NULL 
    """

cur.execute(query)
# -

query = f"""
    CREATE TABLE {username}.intermediate_tb
    STORED AS ORC
    AS SELECT * FROM {username}.intermediate
"""
cur.execute(query)

# ##### c. FInd all the stops which are contain in lausanne region thanks to the previous created tabble and sbb_stops_lausanne_region

# +
query = f"""
    CREATE VIEW {username}.real_time_data
    AS 
    SELECT DISTINCT 
        a.stop_id,
        a.stop_name,
        a.stop_lat,
        a.stop_lon
    FROM 
        {username}.sbb_stops_lausanne_region AS a
    JOIN 
        {username}.intermediate_tb AS b
    ON 
        (lower(a.stop_name)==lower(b.stop_name)
        OR lower(a.stop_id)==lower(b.stop_id))
"""

cur.execute(query)
# -

query = f"""
    CREATE TABLE {username}.real_time_tb
    STORED AS ORC
    AS SELECT * FROM {username}.real_time_data
"""
cur.execute(query)

# ##### d. Validation

# check the result in sbb_stops_lausanne_region 
<<<<<<< HEAD
query = f"SELECT * FROM {username}.real_time_tb LIMIT 5"
pd.read_sql(query, conn)
=======
query = f"SELECT * FROM {username}.real_time_tb"
df_stops_lausanne_real_time  = pd.read_sql(query, conn)
df_stops_lausanne_real_time.head()
>>>>>>> 6c72efb5e6d8dca6859f36b9335586ce1399469d

# %%time
cur.execute(f"""
SELECT COUNT(*) FROM {username}.real_time_tb
""")
cur.fetchall()

# ##### e. Plot

# +
fig = px.scatter_mapbox(
    df_stops_lausanne,
    lat="sbb_stops_lausanne_region.stop_lat",
    lon="sbb_stops_lausanne_region.stop_lon",
    hover_name="sbb_stops_lausanne_region.stop_name",
    zoom=10,
    height=600,
)

fig.add_trace(
    px.scatter_mapbox(
        df_stops_lausanne_real_time,
        lat="real_time_tb.stop_lat",
        lon="real_time_tb.stop_lon",
        hover_name="real_time_tb.stop_name",
        zoom=10,
        height=600,
    ).data[0]
)

fig.data[0].marker.color = "red"
fig.data[0].marker.size = 10
fig.data[1].marker.color = "green"
fig.data[1].marker.size = 10

fig.data[0].name = "All stops"
fig.data[1].name = "Stops with real time data"

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(showlegend=True)
fig.update_layout(title="Stops in Lausanne region with real time data (green) and without (red)")


fig.show()
# -

# ### e) Find stops that are within walking distances of each other - 10/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the pair of stops that are within _500m_ of each other.
# * Save the results in table _{username}.sbb_stops_to_stops_lausanne_region_
# * Validation: you should find around $4500\pm 100$ stops
# * Note: you must complete c) first.

# ##### a. Select stop id where distance is below 500m and insert in sbb_stop_to_stop_lausanne_region 

# +
query = f""" 
    WITH nearby_stops AS (
        SELECT DISTINCT
            stop_id_a,
            stop_id_b,
            distance
        FROM 
            {username}.sbb_stop_to_stop_lausanne_region
        WHERE 
            distance < 500 AND 
            stop_id_a != stop_id_b
            
    )

INSERT OVERWRITE TABLE {username}.sbb_stop_to_stop_lausanne_region 
SELECT * FROM nearby_stops
"""

cur.execute(query)
# -

# ##### b. Validation

# %%time
cur.execute(f"""
SELECT COUNT(*) FROM {username}.sbb_stop_to_stop_lausanne_region
""")
cur.fetchall()

# ### f) Finds the _stop times_ in Lausanne region - 10/50
#
# * Find the stop times that contain only trips (trip_id) that occure on Mondays (per _com490.sbb_orc_calendar_) and only at stops found in the Lausanne region
# * Save the results in the table _{username}.sbb_stop_times_lausanne_region_
# * Validation: you should find around $300K\pm 10K$ trip_id, stop_id pairs.
# * Note: you must complete c) first.

# ##### a. Check if {default_db}.sbb_orc_calendar_ not empty

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_trips LIMIT 5"
pd.read_sql(query, conn)

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_stops LIMIT 5"
pd.read_sql(query, conn)

# check if sbb_orc_stops is not empty
query = f"SELECT * FROM {default_db}.sbb_orc_calendar LIMIT 5"
pd.read_sql(query, conn)

# ##### b. Selection of the trip ID that occurs on monday, creation of temp table

# +
query = f"""
    CREATE TABLE IF NOT EXISTS {username}.service_id_monday
    AS
    SELECT DISTINCT 
        a.service_id
    FROM 
        {default_db}.sbb_orc_calendar as a
    WHERE 
       a.monday
"""

cur.execute(query)
# -

query = f"SELECT * FROM {username}.service_id_monday  LIMIT 5"
pd.read_sql(query, conn)

# +
query = f"""
    CREATE TABLE IF NOT EXISTS {username}.temp_trip_monday 
    AS
    SELECT DISTINCT 
        a.trip_id
    FROM 
        {default_db}.sbb_orc_trips AS a
    WHERE 
        a.service_id IN (
            SELECT 
                service_id 
            FROM 
                {username}.service_id_monday
            )
"""

cur.execute(query)
# -

# check the content of the table 
query = f"SELECT * FROM {username}.temp_trip_monday  LIMIT 5"
pd.read_sql(query, conn)

query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stop_times_lausanne_region
    SELECT
        a.trip_id,
        a.stop_id,
        a.departure_time,
        a.arrival_time
    FROM 
        {username}.sbb_stop_times_lausanne_region as a 
    WHERE 
        a.trip_id IN (
            SELECT 
                trip_id 
            FROM 
                {username}.temp_trip_monday
            )
"""
cur.execute(query)

query = f"""
    INSERT OVERWRITE TABLE {username}.sbb_stop_times_lausanne_region
    SELECT
        a.trip_id,
        a.stop_id,
        a.departure_time,
        a.arrival_time
    FROM 
        {username}.sbb_stop_times_lausanne_region AS a 
    WHERE
        a.stop_id IN (
            SELECT 
                stop_id 
            FROM 
                {username}.sbb_stops_lausanne_region 
            )
"""
cur.execute(query)

# check the content of the table 
query = f"SELECT * FROM {username}.sbb_stop_times_lausanne_region  LIMIT 5"
pd.read_sql(query, conn)

# ##### d. Validation

# %%time
cur.execute(f"""
SELECT COUNT(*) FROM {username}.sbb_stop_times_lausanne_region
""")
cur.fetchall()

# ### g) Design considerations - 2/50
#
# We aim to leverage the findings from e) to suggest an optimal public transport route between two specified locations at a designated time on any given day (not only Monday). Running a query on all data for the week would be wasteful. Considering that travel within the lausanne region should ideally be completed within a 2-hour timeframe, could you advise on enhancing the efficiency of queries made to the _{username}.sbb_stop_times_lausanne_region_ table?
#
# You only need to outline the DDL and DQL commands that you would use to define the table(s) and query the content. Bonus points (+2) will be awarded for implementing the solution.



# ### h) Isochrone Map - 15/50

# Note: This question is open-ended, and credits will be allocated based on the quality of both the proposed algorithm and its implementation. You will receive credits for proposing a robust algorithm, even if you do not carry out the implementation.
#
# Moreover, it is not mandatory to utilize Hive for addressing this question; plain Python is sufficient. You are free to employ any Python package you deem necessary. However, ensure that you include it in either the requirements.txt or the environment.yml file so that we remember to install them.

# **Question**:
# * Given a time of day (always on Monday) and a starting point in Lausanne area.
# * Propose a routing algorithm (such as Bellman-Ford, Dijkstra, A-star, etc.) that leverages the previously created tables to estimate the shortest time required to reach each stop within the Lausanne region using public transport.
# * Visualize the outcomes through a heatmap (e.g., utilizing Plotly), where the color of each stop varies based on the estimated travel time from the specified starting point. For instance, depict shorter durations in green, transitioning to red for durations exceeding an hour. See [example](https://github.com/Samweli/isochrones/blob/master/docs/assets/img/examples/isochrone.png).
# * Hints:
#     - Focus solely on scenarios where walking between stops is not permitted. Once an algorithm is established, walking can optionally be incorporated, assuming a walking speed of 50 meters per minute. Walking being optional, bonus points (+2) will be awarded for implementing it. 
#     - If walking is not considered, a journey consists of a sequence of stop_id separated by trip_id in chronological order, e.g. _stop-1_, _trip-1_, _stop-2_, _trip-2_, ..., _stop-n_.
#     - Connections between consicutive stops and trips can only happen at predetermined time: each trip-id, stop-id pair should be unique at occure at a specific time on any given day, and you cannot go back in time to catch an earlier connection after you arrive at a stop (you must take an earlier connection).

stop_times = pd.read_csv("hdfs://iccluster067.iccluster.epfl.ch:8020/data/sbb/share/stop_times_df.csv")
stop_to_stop = pd.read_csv("hdfs://iccluster067.iccluster.epfl.ch:8020/data/sbb/share/stop_to_stop_df.csv")
stops = pd.read_csv("hdfs://iccluster067.iccluster.epfl.ch:8020/data/sbb/share/stops_df.csv")


def fix_time_format(time_str):
    try:
        parts = time_str.split(' ')
        days = int(parts[0])
        hours, minutes, seconds = map(int, parts[2].split(':'))
        return pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    except:
        return pd.NaT


# +
def assign_arrival_id(row, df):
    if row.name == len(df) -1:
        return row['stop_id']
    elif row['next_stop_id'] is None or row['trip_id'] != df.loc[row.name + 1, 'trip_id']:
        return row['stop_id']
    else:
        return row['next_stop_id']

def assign_arrival_time(row, df):
    if row.name == len(df) -1:
        return row['arrival_time_dt']
    elif row['tmp_arrival_time_dt'] is None or row['trip_id'] != df.loc[row.name + 1, 'trip_id']:
        return row['arrival_time_dt']
    else:
        return row['tmp_arrival_time_dt']

# Departure = current stop / Arrival = future stop
# Formatting and renaming
stop_times['departure_time_dt'] = stop_times['arrival_time_dt'].apply(lambda x: fix_time_format(x))
#stop_times['departure_time_dt'] =  pd.to_timedelta(stop_times['departure_time_dt'])
#stop_times['departure_time_dt'] = stop_times['departure_time_dt'].apply(lambda x: fix_time_format(x))

# adding information of next trip
stop_times['next_stop_id'] = stop_times['stop_id'].shift(-1)
stop_times['tmp_arrival_time_dt'] = stop_times['arrival_time_dt'].shift(-1)

stop_times['arrival_id'] = stop_times.apply(assign_arrival_id, axis=1, df=stop_times)
stop_times['arrival_time_dt'] = stop_times.apply(assign_arrival_time, axis=1, df=stop_times)

# renaming
stop_times['departure_id'] = stop_times['stop_id']

# Drop the 'next_stop_id' column
stop_times.drop(columns=['tmp_arrival_time_dt', 'arrival_time', 'departure_time', 'stop_id', 'next_stop_id'], inplace=True)
# reorder columns
stop_times = stop_times[['trip_id', 'departure_id', 'departure_time_dt', 'arrival_id', 'arrival_time_dt']]
# Drop trips with same start and end
stop_times = stop_times[~(stop_times['departure_id'] == stop_times['arrival_id'])].reset_index(drop=True)
<<<<<<< HEAD
=======
# -

# not used
# stop_times[stop_times['trip_id'].apply(lambda x: '100.TA.96-287-j24-1.12.R' in x)].head()
>>>>>>> 6c72efb5e6d8dca6859f36b9335586ce1399469d

from math import radians, cos, sin, asin, sqrt
def calculate_distance_geodata(lat1, lon1, lat2, lon2):
    # from https://www.geeksforgeeks.org/program-distance-two-points-earth/
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)


def keep_unique_stops_only(df):
    if len(df)>=1:
        mask = df.duplicated(subset=['arrival_id'])
        return df[~mask].reset_index(drop=True)
    return df


def get_stop_geodata(stop_id, df=stops):
    tmp = df[df['stop_id'] == stop_id]
    if len(tmp)>0:
        return tmp.iloc[0,[2,3]]
    else:
        return [0,0]


def calculate_distance_stops(stop1, stop2):
    lat1, lon1 = get_stop_geodata(stop1)
    lat2, lon2 = get_stop_geodata(stop2)
    return calculate_distance_geodata(lat1, lon1, lat2, lon2)


def calculate_and_weight_distances(dataframe, end_id, weighting_modifier=1.4):
    if len(dataframe) >=1:
        dataframe['distance'] = dataframe['arrival_id'].apply(lambda x: calculate_distance_stops(x, end_id))
        # weight with walking speed 1.4m/s
        dataframe['weighted_distance'] = dataframe['distance'] 
        # or: dataframe.apply(lambda x: x['distance'] * 1000 + 1.4 * x['timedifference'] *60, axis=1)
        dataframe.sort_values(by='weighted_distance', inplace=True, ignore_index=True)
    return dataframe


def time_diff(time1, time2):
    diff = pd.Timedelta(time1) - pd.Timedelta(time2)
    return abs(diff.total_seconds() / 60)


def get_outgoing_connections( start_id, time, lookahead=15, df=stop_times,):

    stop = df[df['departure_id'] == start_id].reset_index(drop=True)
    mask = stop['departure_time_dt'].apply(lambda x: time_diff(x, time) <= lookahead and pd.Timedelta(x) > pd.Timedelta(time))
    stop = stop[mask].reset_index(drop=True)
    if len(stop):
        stop['timedifference'] = stop['departure_time_dt'].apply(lambda x: time_diff(x, time))
        stop.sort_values(by='departure_time_dt')
    return stop


def get_walking_opportunities(start_id, time):
    output = pd.DataFrame()
    stations = stop_to_stop[stop_to_stop['stop_id_a'] == start_id]
    output['departure_id'] = stations['stop_id_a']
    output['trip_id'] = 'walk'
    output['arrival_id'] = stations['stop_id_b']
    output['timedifference'] = stations['distance'].apply(lambda x: round(x/50, 2))
    output['departure_time_dt'] = time
    if len(output)>=1:
        output['arrival_time_dt'] = output.apply(lambda x: pd.Timedelta(minutes = x['timedifference']) + pd.Timedelta(x['departure_time_dt']), axis=1)
    return output


def search_quickest_connection(start_id, end_id, time_start, time_step, max_time, trip_route=None, visited=None):
    connection = (1,'no route found',set())
    if trip_route == None:
        trip_route=[]
    if visited == None:
        visited=set()
    trip_route.append(start_id)
    
    if start_id == end_id:
        return(0, time_diff(time_start, time_step), trip_route)
    elif start_id in visited:
        return (1,'already visited', visited)
    elif time_diff(time_start, time_step) > max_time:
        return (1,'time exceeded', visited)
    
    visited.add(start_id)
    connections = get_outgoing_connections(start_id, time_step, lookahead=15)
    walking = get_walking_opportunities(start_id, time_step)
    directions = pd.concat([connections,walking])
    directions = keep_unique_stops_only(directions)
    directions = calculate_and_weight_distances(directions, end_id)

    for direction in directions.iterrows():
        connection = search_quickest_connection(
            direction[1]['arrival_id'],
            end_id,
            time_start,
            direction[1]['arrival_time_dt'],
            max_time=max_time,
            trip_route=trip_route,
            visited = visited)

        if connection[0] == 0:
            return connection
        visited = visited.union(connection[2])
    return connection
    
    # get all connections we can go from start_id
        # weight connection with distance gained and time passed
    # order connections and pick best
    # recursion until stop_id reached or time difference exceeds 1h
    # if time difference > 1h or no more options:
        #backtrack and take second best option


stops[stops['stop_name'].apply(lambda x: 'gare' in x.lower())]

search_quickest_connection('8592050', '8592050:0:C', pd.Timedelta('0 days 08:11:00'), pd.Timedelta('0 days 08:11:00'), max_time=120)

# +
#run algorithm for all stops in lausanne:
#traveltime = stops['stop_id'].apply(lambda x: search_quickest_connection(x, '8501210',pd.Timedelta('0 days 08:11:00'),pd.Timedelta('0 days 08:11:00'),max_time=60))

#travel_times_lausanne = pd.concat([traveltime, stops['stop_id']], axis=1)
#travel_times_lausanne.columns = ['traveltime', 'stop_id']
#merged = pd.merge(travel_times_lausanne, stops, on='stop_id', how='inner')

#merged.to_csv('travel_times_lausanne_walking.csv', index=False)
# -

# # Plot

import ast

walking = pd.read_csv('travel_times_lausanne_walking.csv')
walking['traveltime'] = walking['traveltime'].apply(lambda x: ast.literal_eval(x))
walking['minutes'] = walking['traveltime'].apply(lambda x: 60 if x[0] else x[1])

# This is the Final heatmap, showing travel times from Lausanne Bourdonette to other stations on Monday 08:11:00:

fig = px.density_mapbox(walking, lat='stop_lat', lon='stop_lon', z='minutes', radius=30,
                        center=dict(lat=46.523266, lon=6.589808), zoom=12,
                        opacity = 0.9,
                        color_continuous_scale = 'curl',
                        mapbox_style="open-street-map",
                        hover_name='stop_name',
                       )
fig.show()
