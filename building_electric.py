from pandas._libs.tslibs import Timestamp
from pandas.core.construction import array
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from building_data import *
import numpy as np
import plotly.express as px

path = 'building_data'
metadata = pd.read_csv(path + '/meta_open.csv')

st.title('This webapp will show analysis of building electricity consumption and use KNN to make predictions for the usage')
st.markdown('### We are using a metadata file that summarizes the office information and weather data.')
st.write(metadata.head())

Timezone_data={'America/New_York','Europe/London','America/Phoenix','America/Chicago','America/Los_Angeles','Europe/Zurich','Asia/Singapore','America/Denver'}
Industry_data={'Education','Government','Commercial Property'}

st.sidebar.subheader('Filters')
timezone = st.sidebar.selectbox("Select a timezone", tuple(Timezone_data))
industry = st.sidebar.selectbox("Select an industry", tuple(Industry_data))

filtered_metadata = metadata.loc[(metadata['timezone']==timezone)&(metadata['industry']==industry)]
st.markdown('### Once filters are applied, we can focus on office name and the weather data. For each site, we will have a dataframe for building usage and a dataframe for weather data')
st.write(filtered_metadata[['uid','newweatherfilename']])

building_file = filtered_metadata['uid'].values[0]+'.csv'
building_data = pd.read_csv(path+'/{}'.format(building_file))

weather_file = filtered_metadata['newweatherfilename'].values[1]
weather_data = pd.read_csv(path+'/{}'.format(weather_file))

st.write(building_data.head())
st.write(weather_data.head())

st.markdown('### Let\'s make the timestamp column a datetime object and then make it the index of the building dataframe. We also have to rename the columns adequately')
building_data['timestamp'] = pd.to_datetime(building_data['timestamp'])
building_data.columns=['timestamp','kWh']
building_data = building_data.set_index('timestamp')
building_data = building_data.resample('H').mean()

st.write(building_data.head())

st.markdown('### Now the same will be done to the weather data. For simplicity, we will only focus on temperature and humidity.')    
weather_data= weather_data[['timestamp','TemperatureC','Humidity']]
weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
weather_data = weather_data.set_index('timestamp')
weather_data = weather_data.resample('H').mean()

st.write(weather_data.head())


st.write('Now let\'s do a left join of building (left table) and weather data.')
clean_merged_data = pd.merge(building_data,weather_data,how='left',on='timestamp')
clean_merged_data = clean_merged_data.fillna(method='bfill')

st.write(clean_merged_data.head())

st.write('Now we need to normalize the data. Normalization helps ensure that every numerical column has the same range of values so it is easier to make a distribution.')

normalized_df = (clean_merged_data - clean_merged_data.mean()) / (clean_merged_data.std())
st.write(normalized_df.head())

y_mean = clean_merged_data['kWh'].mean()
y_std = clean_merged_data['kWh'].std()

def revert_label_value(pred):
    return (pred*y_std+y_mean)

st.write('Now we shall split the data into training and test sets. Since we have dates, the split will consist of splitting the dates before certain dates. You can change these numbers on the sidebar.')

"""
Having trouble designing a slider or other widget that sets a date filter to split the data
"""
# max_date = clean_merged_data.dt
# min_date = clean_merged_data.index.min()
# select_date = st.sidebar.date_input('Choose date range to split data: ',(min_date,max_date))
train_set = normalized_df.truncate(before='2014-01-01', after='2014-12-31')
test_set = normalized_df.truncate(before='2015-01-01',after='2015-12-31')

st.write('Train set: ')
st.write(train_set.head())
st.write('Test set: ')
st.write(test_set.head())

# Select labels
train_label = train_set['kWh']
test_label = test_set['kWh']

#Encode train features
x_train=pd.get_dummies(train_set.index.hour)
y_train=pd.get_dummies(train_set.index.dayofweek)
t_train=pd.DataFrame(train_set['TemperatureC'].values)
h_train=pd.DataFrame(train_set['Humidity'].values)
train_features = pd.concat([x_train,y_train,t_train,h_train],axis=1).dropna()

#Encode test features 
x_test=pd.get_dummies(test_set.index.hour)
y_test=pd.get_dummies(test_set.index.dayofweek)
t_test=pd.DataFrame(test_set['TemperatureC'].values)
h_test=pd.DataFrame(test_set['Humidity'].values)
test_features = pd.concat([x_test,y_test,t_test,h_test],axis=1).dropna()


choose_k = st.sidebar.selectbox("Choose number of neighbors: ", (1,2,3,4,5,6))

knnreg = KNeighborsRegressor(n_neighbors=choose_k).fit(np.array(train_features), np.array(train_label.values))
predicted = knnreg.predict(np.array(test_features))


# Convert predict to Dataframe
predicted = pd.DataFrame(predicted,index=test_set.index)
#Give name to the predict column
predicted.columns=['kWh']
#revert back the distribution to the original distribution
predicted = predicted['kWh'].apply(lambda x: revert_label_value(x))
#Test
#st.write(predicted)
#revert back the distribution to the original distribution
test_set['kWh'] = test_set['kWh'].apply(lambda x: revert_label_value(x))
#Test
#test_set['kWh']

# merge the actual test set with the predicted one
actual_v_predicted = pd.merge(test_set, predicted,how='left',on='timestamp')
#name the columns 
actual_v_predicted.columns = ['kWh_Actual','Temp_Actual','Humid_Actual','kWh_Predicted']

#comparison set
actual_v_predicted_plot = actual_v_predicted[['kWh_Actual','kWh_Predicted']]
st.markdown('### Actual vs prediced electricity consumption')
actual_v_predicted_plot.info()


#plot set
fig = px.scatter(x=actual_v_predicted_plot['kWh_Actual'],y=actual_v_predicted_plot['kWh_Predicted'])
#st.set_option('deprecation.showPyplotGlobalUse',False) # This is needed to avoid raising a warning for calling pyplot with no arguments
st.plotly_chart(fig)

# actual_v_predicted_plot['kWh_Actual'] = array.reshape(actual_v[_pedicted_plo,-1])
fitted = knnreg.fit(actual_v_predicted_plot[['kWh_Actual']], actual_v_predicted[['kWh_Predicted']])
score = knnreg.score(actual_v_predicted_plot[['kWh_Actual']], actual_v_predicted[['kWh_Predicted']])

st.write('The coefficient of determination for this prediction is: ' + str(round(score,3)))