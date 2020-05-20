# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:39:42 2019

@author: swaraj 
"""

# Importing  the necceasry libraries we will need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

import math

sns.set(font_scale=2)
cmap = sns.diverging_palette(220, 10, as_cmap=True) # one of the many color mappings

# Read in the crime data from the each single year CSV file
df1 = pd.read_csv('Crime_Incidents_in_2012.csv')
df2 = pd.read_csv('Crime_Incidents_in_2013.csv')
df3 = pd.read_csv('Crime_Incidents_in_2014.csv')
df4 = pd.read_csv('Crime_Incidents_in_2015.csv')
df5 = pd.read_csv('Crime_Incidents_in_2016.csv')
df6 = pd.read_csv('Crime_Incidents_in_2017.csv')
df7 = pd.read_csv('Crime_Incidents_in_2018.csv')

#printing the column names
COLUMN_NAMES = ['X', 'Y', 'CCN', 'REPORT_DAT', 'SHIFT', 'METHOD', 'OFFENSE', 'BLOCK', 'XBLOCK',
                 'YBLOCK', 'WARD', 'ANC', 'DISTRICT', 'PSA','NEIGHBORHOOD_CLUSTER', 'BLOCK_GROUP', 'CENSUS_TRACT',
                 'VOTING_PRECINCT', 'LATITUDE', 'LONGITUDE', 'BID', 'START_DATE','END_DATE', 'OBJECTID']
df = pd.DataFrame(columns=COLUMN_NAMES)
print(df.columns)   # checking consistency of column names
print(df1.columns)
print(df2.columns)
print(df3.columns)
print(df4.columns)
print(df5.columns)
print(df6.columns)
print(df7.columns)
df =pd.concat([df1,df2, df3,df4,df5,df6,df7],ignore_index =True)  # concatinating the  DataFrames together
print(df.info())#checking the data in the dataframes
print(df)

#storing the data to a file named dc_crime_project
df.to_csv('dcrime.csv')

# setting the visual
sns.set(font_scale=2)
cmap = sns.diverging_palette(220, 10, as_cmap=True) # one of the many color mappings
sns.set_style('whitegrid')
#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline


df = pd.read_csv('dcrime.csv')
test = pd.read_csv('Crime_Incidents_in_2018.csv')
ward_data =pd.read_csv('warddata.csv')
#hp = pd.read_csv('house_sale.csv')

print (df.duplicated().sum())


print(df.shape)
print(df.head(2))
print(ward_data.shape)
print(ward_data.head(2))


print(df.isnull().sum().sort_values(ascending=False))


# checking the null values of the df and sorting them as percentage of the DF shape
print((df.isnull().sum()/df.shape[0]).sort_values(ascending=False))

print(ward_data.isnull().sum()/len(ward_data))


print('***************************************************')
print( "---***<<< Duty Shift: variable = 'SHIFT' >>>***--")
var_count = df.groupby('SHIFT')
print( var_count.CCN.count())
print('***************************************************')

print('***************************************************')
print("---==< Crimes committed per police district: variable = 'DISTRICT' >==---")
var_count = df.groupby('DISTRICT')
print(var_count.CCN.count())
print('***************************************************')


print('***************************************************')
print("---==< Crimes|OFFENSES committed per BLOCK: variable = 'BLOCK' >==---")
print(df.groupby('BLOCK').OFFENSE.value_counts())
print('***************************************************')


print('***************************************************')
print('--====< Example Record from Data Set>============')
print(df.iloc[1234])
print('***************************************************')


print('***************************************************')
print(df.head())

#print (np.count_nonzero(df['BLOCK'].unique()))
df.drop('BLOCK', axis=1, inplace=True)
print(df.head())
#print (np.count_nonzero(df['BLOCK_GROUP'].unique()))
df.drop('BLOCK_GROUP', axis=1, inplace=True)
print('***************************************************')



# if END_DATE is NaN, then use START_DATE
df['END_DATE'].fillna(df['START_DATE'], inplace=True)

# if VOTING_PRECINCT is NaN, then set it to 0
df['VOTING_PRECINCT'].fillna(0, inplace=True)

# if NEIGHBORHOOD_CLUSTER is NaN, then set it to 0
df['NEIGHBORHOOD_CLUSTER'].fillna(0, inplace=True)

# if CENSUS_TRACT is NaN, then set it to 0
df['CENSUS_TRACT'].fillna(0, inplace=True)
# if WARD is NaN, then use ANC
# df['WARD'].fillna(df['ANC'], inplace=True)
# if WARD is NaN, then set it to 0
df['WARD'].fillna(0, inplace=True)



#  Create a dataframe that holds the central location of each Police Service Area (PSA)
#  The PSAs: according to the information from the DC metropolitan ionformation and opendata DC
# PSA's are smaller than the Police Districts, so we should have better accuracy in identifying the associated PSA
#  The PSA ID contains the District ID, so we can impute the District from the PSA
psa_loc = pd.DataFrame(df[['PSA','XBLOCK','YBLOCK']].groupby('PSA').median())
#  ---==< Estimate PSA membership based on proximity to each area's centroid >==---
def nearbyPSA(nPSA,dX,dY):
    # Default to the current PSA ID
    nearbyPSA = nPSA
    
    # Only operate on missing IDs
    if (pd.isnull(nPSA)):
        minDist = 9e99  # Set the initial closest distance to be a large value
        nearbyPSA = 0
        
        # Loop through the records in the psa_loc dataframe
        for PSA_ID, PSA in psa_loc.iterrows():
            # Calculate the distance between the report and the current PSA using the Eucleadian distance
            thisDist = math.sqrt((dX - PSA['XBLOCK'])**2 + (dY - PSA['YBLOCK'])**2)
            
            # If this distance is smaller than the current minimum distance, update the minimum distance
            if (thisDist < minDist):
                minDist = thisDist # Replace the minimum distance with the current distance
                nearbyPSA = PSA_ID # Remember which PSA this is related to
                
    # Return the ID for the closest PSA
    return [nearbyPSA, int(nearbyPSA / 100)]

#  Impute the missing PSA
df['PSA_ID'] = 0
df['DistrictID'] = 0
df[['PSA_ID','DistrictID']] = list(map(nearbyPSA,df['PSA'],df['XBLOCK'],df['YBLOCK']))

#  Display the results of this imputation method.
print('***************************************************')
print(df[['PSA','DISTRICT','PSA_ID','DistrictID']][df['PSA'].isnull()])
print('***************************************************')





# do we have duplicate rows?
# the great thing is that i took care of them while concatenating the different values in the initial Data setup
print(df.duplicated().sum())


# strip 'Precinct ' from VOTING_PRECINCT values
# http://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].apply(str).map(lambda x: x.lstrip('Precinct '))

# strip 'Cluster ' from NEIGHBORHOOD_CLUSTER values
df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].apply(str).map(lambda x: x.lstrip('Cluster '))



print('***************************************************')
# convert REPORT_DAT to datetime
df['REPORT_DAT'] = pd.to_datetime(df['REPORT_DAT'])

# convert SHIFT to int
shift_mapping = {'day':1, 'evening':2, 'midnight':3}
df['SHIFT_Code'] = df['SHIFT'].str.lower().map(shift_mapping).astype('category')

# convert OFFENSE to numeric
# Python for Data Analysis, pg. 279
offense_mapping = {'theft/other':1, 'theft f/auto':2, 'burglary':3, 'assault w/dangerous weapon':4, 'robbery':5,
                  'motor vehicle theft':6, 'homicide':7, 'sex abuse':8, 'arson':9}
df['OFFENSE_Code'] = df['OFFENSE'].str.lower().map(offense_mapping).astype('category')
df['OFFENSE'] = df['OFFENSE'].str.replace('DANGEROUS WEAPON', 'DW')

# convert METHOD to numeric
method_mapping = {'others':1, 'gun':2, 'knife':3}
df['METHOD_Code'] = df['METHOD'].str.lower().map(method_mapping).astype('category')

# convert DISTRICT to numeric
df['DistrictID'] = df['DistrictID'].astype(np.int64)

# convert PSA to numeric
df['PSA_ID'] = df['PSA_ID'].astype(np.int64)

# convert WARD to numeric
df['WARD'] = df['WARD'].astype(np.int64)

# convert ANC to numeric
anc_mapping = {'1B':12, '1D':14, '1A':11, '1C':13, '6E':65, '4C':43, '5E':55, '2B':22, '2D':24, '2F':26, '2C':23,
       '2E':25, '2A':21, '3C':33, '3E':35, '3B':32, '3D':34, '3F':36, '3G':37, '4A':41, '4B':42, '4D':44,
       '5A':51, '5D':54, '5C':53, '5B':52, '6A':61, '6C':63, '6B':62, '6D':64, '7D':74, '7C':73, '7E':75,
       '7B':72, '7F':76, '8A':81, '8B':82, '8C':83, '8D':84, '8E':85}
df['ANC'] = df['ANC'].map(anc_mapping).astype('category')

# convert NEIGHBORHOOD_CLUSTER to numeric
df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(np.int64)

# convert CENSUS_TRACT to numeric
df['CENSUS_TRACT'] = df['CENSUS_TRACT'].astype(np.int64)

# convert VOTING_PRECINCT to numeric
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(np.int64)

# convert CCN to numeric
df['CCN'] = df['CCN'].astype(np.int64)

# convert XBLOCK, YBLOCK to numeric
df['XBLOCK'] = df['XBLOCK'].astype(np.float64)
df['YBLOCK'] = df['YBLOCK'].astype(np.float64)

# convert START_DATE, END_DATE to dateime
df['START_DATE'] = pd.to_datetime(df['START_DATE'])
df['END_DATE'] = pd.to_datetime(df['END_DATE'])
print('***************************************************')
print(df.info())
print()
print(df.iloc[1234])
print('***************************************************')


print('***************************************************')
print('---======< Dealing with outliers>=======-------')
print(df['START_DATE'][df['START_DATE']<'1/1/2011'].count())
print(sorted(df['START_DATE'][df['START_DATE']<'1/1/2011'])[:10])
print('***************************************************')

print('***************************************************')
print('---======< Dealing with outliers>=======-------')
print(df['START_DATE'][df['START_DATE']<'1/1/2012'].count())
print(sorted(df['START_DATE'][df['START_DATE']<'1/1/2012'])[:10])
print('***************************************************')

print('***************************************************')
print('---======< Dealing with outliers>=======-------')
print(df['START_DATE'][df['START_DATE']<'1/1/2013'].count())
print(sorted(df['START_DATE'][df['START_DATE']<'1/1/2013'])[:10])
print('***************************************************')

# creating feature for crime type 1 = Violent, 2 = non violent
violent_offense = [4, 5, 7, 8]
df['CRIME_TYPE'] = np.where(df['OFFENSE_Code'].isin(violent_offense), 1, 2)
df['CRIME_TYPE'] = df['CRIME_TYPE'].astype('category')



# create age of crime END_DATE - START_DATE in seconds
df['AGE'] = (df['END_DATE'] - df['START_DATE'])/np.timedelta64(1, 's')




# create new feature TIME_TO_REPORT to indicate the timespan between the latest time the crime was committed and
# the time it was reported to the MPD. that is  time it took from crime to report it, REPORT_DAT - END_DATE in seconds
df['TIME_TO_REPORT'] = (df['REPORT_DAT'] - df['END_DATE'])/np.timedelta64(1, 's')
df['DATE'] = pd.to_datetime(df['END_DATE'], format = '%d/%m/%Y %H:%M:%S')


def date_separate(df):
    """ function that separates the sepecific interest days,weeks from the END_DATE"""
    df = df.copy()
    df['Year'] = pd.DatetimeIndex(df['DATE']).year
    df['Month'] = pd.DatetimeIndex(df['DATE']).month
    df['Day'] = pd.DatetimeIndex(df['DATE']).day
    df['hour'] = pd.DatetimeIndex(df['DATE']).hour
    df['dayofyear'] =pd.DatetimeIndex(df['DATE']).dayofyear
    df['week'] =pd.DatetimeIndex(df['DATE']).week
    df['weekofyear'] =pd.DatetimeIndex(df['DATE']).weekofyear
    df['dayofweek'] =pd.DatetimeIndex(df['DATE']).dayofweek
    df['weekday'] =pd.DatetimeIndex(df['DATE']).weekday
    df['quarter'] =pd.DatetimeIndex(df['DATE']).quarter
    return df



# here i have created an additional columns that helps us in analyzing the timing of the crimes happeing either by day
# week, month, year as well as the quarters and weekdays and week ends.
df['Year'] = pd.DatetimeIndex(df['DATE']).year
df['Month'] = pd.DatetimeIndex(df['DATE']).month
df['Day'] = pd.DatetimeIndex(df['DATE']).day
df['hour'] = pd.DatetimeIndex(df['DATE']).hour
df['dayofyear'] =pd.DatetimeIndex(df['DATE']).dayofyear
df['week'] =pd.DatetimeIndex(df['DATE']).week
df['weekofyear'] =pd.DatetimeIndex(df['DATE']).weekofyear
df['dayofweek'] =pd.DatetimeIndex(df['DATE']).dayofweek
df['weekday'] =pd.DatetimeIndex(df['DATE']).weekday
df['quarter'] =pd.DatetimeIndex(df['DATE']).quarter


print('***************************************************')
print('---<Examine the frequency of types of crimes/offenses>--======')
print('    __________________________________________________')
print(' Total Offenses - Count')
total_crime = df.CCN.count()
print (total_crime)
print('')
crime_rate = df.groupby('OFFENSE')
print('_________________________________')
print (' Offense Type - Count')
print('_________________________________')
print (crime_rate.CCN.count())
print ('')
print('_________________________________')
print (' Offense as percentage of total ')
print ('_______________________________')
print (crime_rate.CCN.count() / total_crime * 100.0)
print()
print ('___________________________')
print (' Offense Rate per 100,000 ')
print ('__________________________')
print (crime_rate.CCN.count() / 6.93972 )
print()
print ('_____________________________________')
print (' Odds of being a victim - by offense')
print ('____________________________________')
print (crime_rate.CCN.count() / 693972)

print('***************************************************')

print('-<Examine the frequency of types of crimes/offenses by Method>===')
print ('_____________________________________')
print(' Total Methods - Count') 
print ('_____________________________________')
print(df.CCN.count())
print ('')

method_rate = df.groupby('METHOD')
print ('_____________________')
print ('Method Type - Count')
print ('____________________')
print(method_rate.CCN.count())
print()
print ('______________________________')
print (' Method as percentage of total ')
print ('_____________________________')
print (method_rate.CCN.count() / total_crime * 100.0)

print ('_________________________')
print (' Method Rate per 100,000 ')
print ('_________________________')
print( method_rate.CCN.count() / 6.93972)
print ('___________________________________')
print (' Odds of being a victim - by method')
print ('_______________________________')
print (method_rate.CCN.count() / 693972)
print('***************************************************')


print('***************************************************')
print('-<Examine the age of  crimes/offenses by Time >===')

# hours
temp = df['AGE'] / 3600

print( temp.describe())
print('') 

# exclude temp two std away from mean
#http://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-dataframe
print (temp[~(np.abs(temp - temp.mean())>(2*temp.std()))].describe())
print('***************************************************')


print('***************************************************')
print('-<Examine the frequency of time it takes to report a of crimes/offenses >===')
# hours
temp = df['TIME_TO_REPORT'] / 3600

print(temp.describe())
print ('-----------------------')

# excluding temp two std away from mean
print (temp[~(np.abs(temp - temp.mean())>(2*temp.std()))].describe())
print('***************************************************')

#prining the data to another csv file
df.to_csv('dc_crime2.csv')



# setting the visual 
sns.set(font_scale=2)
cmap = sns.diverging_palette(220, 10, as_cmap=True) # one of the many color mappings
sns.set_style('whitegrid')

#reading the csv in a df file
df = pd.read_csv('dc_crime2.csv')
ward_data =pd.read_csv('warddata.csv')

print("---< categorized  offences  >----")
temp_var = df[['SHIFT', 'CRIME_TYPE']]
plt.figure(figsize=(8, 5))
sns.countplot(x='SHIFT', hue='CRIME_TYPE', data=temp_var)

#  1 = Violent, 2 = nonviolent
#plotting a box plot
sns.boxplot(x='SHIFT' ,y='hour' ,data=df , palette='winter_r')

temp_var = df[['DistrictID', 'METHOD']]
temp_var1 = df[['DistrictID', 'METHOD']]
temp_var1 = temp_var1[temp_var1['METHOD'] != "OTHERS"]

plt.figure(figsize=(10, 5))
sns.countplot(x='DistrictID', hue='METHOD', data=temp_var)
plt.figure(figsize=(10, 5))
sns.countplot(x='DistrictID', hue='METHOD', data=temp_var1)

#  1 = Others, 2 = Gun, 3 =  Knife
#count plot using seaborn
temp_var = df[['DistrictID', 'CRIME_TYPE']]
plt.figure(figsize=(15, 9))
sns.countplot(x='DistrictID', hue='CRIME_TYPE', data=temp_var)

temp_var = pd.crosstab(df.DistrictID, df.OFFENSE)
#print temp_vardf

temp_var.plot(kind='bar', stacked=True, figsize=(25, 15))
plt.legend(loc=9, bbox_to_anchor=(.5, -.2), ncol=9)


temp_var = df[['Month', 'CRIME_TYPE']]
plt.figure(figsize=(20, 10))
sns.countplot(x='Month', hue='CRIME_TYPE', data=temp_var)

temp_var = df[['weekofyear', 'CRIME_TYPE']]
plt.figure(figsize=(50, 45))
sns.countplot(x= 'weekofyear', hue='CRIME_TYPE', data=temp_var)

df.columns

temp_var = df[['quarter', 'CRIME_TYPE']]
plt.figure(figsize=(50, 45))
sns.countplot(x= 'quarter', hue='CRIME_TYPE', data=temp_var)

def percentConv(x):
    return x / float(sum(x))

print( pd.crosstab(df.OFFENSE, df.SHIFT).apply(percentConv, axis=1))

print (pd.crosstab(df.DistrictID, df.CRIME_TYPE, margins=True))
print ('-----------------------------------------------------')
print (pd.crosstab(df.DistrictID, df.CRIME_TYPE).apply(percentConv, axis=1))

print('---------------------------------------------------------------') 
print ('Percentage of Crime in each District')
print("***************************************************")
temp_var_total = df.DistrictID.value_counts().sum()
print(df.DistrictID.value_counts() / temp_var_total)

print("***************************************************")
print (pd.crosstab(df.OFFENSE, df.METHOD, margins=True))
print("***************************************************")

print("***************************************************")
print(pd.crosstab(df.OFFENSE, df.METHOD).apply(percentConv, axis=1))
print("***************************************************")

print("******************************************************************************************************")
# 0 = Monday, 6 = Sunday

# set index to the start of the crime
g = df.dayofweek
print( pd.crosstab(df.OFFENSE, g, colnames=['dayofweek'], margins=True))
print("******************************************************************************************************")
print(pd.crosstab(df.OFFENSE, g, colnames=['dayofweek']).apply(percentConv, axis=1))
print("******************************************************************************************************")

print("***************************************************Reporting  times per offense type***************************************************")
# Set up a wide chart so we can see the separation between offenses
plt.figure(figsize=(30,10))

# Default font was too small to make out the Offense, so scale it
sns.set(font_scale=2)

# Create a subset with data from 0 to 24 hours response times
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]

# Create the box plot - report the response time in hours instead of seconds.  Group by Offense, and color by Shift
sns.boxplot(x=plt_test.OFFENSE, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)

# Move the legend out of the way
plt.legend(loc='upper right')

print("*************************************************** Reporting times  when a Dangerous Weapon is involved***************************************************")
# Set up a wide chart so we can see the separation between offenses
plt.figure(figsize=(30,10))

# Default font was too small to make out the labels, so scale it
sns.set(font_scale=2)

# Create a subset with data from 0 to 24 hours response times
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]

# Create the box plot - report the response time in hours instead of seconds.  Group by Method, and color by Shift
sns.boxplot(x=plt_test.METHOD, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)

# Move the legend out of the way
plt.legend(loc='upper right')

print("***************************************************Reporting times  per District and Shift***************************************************")
plt.figure(figsize=(40,10))
sns.set(font_scale=3)
#
# Filter data to just within a 24-hour period
plt_test = df[df.TIME_TO_REPORT < 86400][df.TIME_TO_REPORT > 0]

# Display response time in hours (3600 seconds)
sns.boxplot(x=plt_test.DistrictID, y=plt_test.TIME_TO_REPORT/3600.0, hue=plt_test.SHIFT)
plt.legend(loc='upper right')


print("***************************************************Plot of violent crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper left')
plt.show()
print("***************************************************Plot of violent crimes***************************************************")

print("***************************************************Plot of violent crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************violent crimes in Washington DC ***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("*************************************************** violent crimes[ROBBERY] in Washington DC ***************************************************")

print("*************************************************** Plot of violent crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ROBBERY'], df['LATITUDE'][df['OFFENSE']=='ROBBERY'], s=50, alpha=0.3, color=[0.0,1.0,1.0], lw=0, label='Robbery')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************violent crimes in Washington DC ***************************************************")


print("***************************************************Plot of violent[ASSAULT W/DW] crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)

plt.scatter(df['LONGITUDE'][df['OFFENSE']=='ASSAULT W/DW'], df['LATITUDE'][df['OFFENSE']=='ASSAULT W/DW'], s=50, alpha=0.5, color='g', lw=0, label='Assault')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************violent crimes[ASSAULT W/DW] in Washington DC ***************************************************")

print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes ***************************************************")

print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes***************************************************")

print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes***************************************************")
plt.figure(figsize=(15, 15))
sns.set(font_scale=2)
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='SEX ABUSE'], df['LATITUDE'][df['OFFENSE']=='SEX ABUSE'], s=50, alpha=0.3, color='b', lw=0, label='Sex Abuse')
plt.scatter(df['LONGITUDE'][df['OFFENSE']=='HOMICIDE'], df['LATITUDE'][df['OFFENSE']=='HOMICIDE'], s=50, alpha=0.3, color='r', lw=0, label='Homicide')
plt.scatter(-77.03654,38.89722,s=60, color=[1,0,1], lw=1, label='White House')
plt.scatter(-77.00937,38.88968,s=60, color=[0,0,0], lw=1, label='U.S. Capitol')
plt.legend(loc='upper right')
plt.show()
print("***************************************************Plot of violent[SEX ABUSE and HOMICIDE] crimes***************************************************")

#defing the correlation function for multiple use
def plot_corr(df, size = 17):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax, shrink = .8)
    ax.grid(True)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 'vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    #we are plotiing the correlation of data
plot_corr(df)
corrMatrix = df.corr()
corrMatrix.to_csv('dc_crime1corr.csv')
#plotting the pie charts
df.to_csv('dc_crime3.csv')
df.SHIFT.value_counts()

#pie chart
labels = 'Morning', 'Evening', 'Night'
#sizes = []
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(df.SHIFT.value_counts(), labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

