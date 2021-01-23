# Dependencies
import warnings
warnings.filterwarnings('ignore')
import sys 
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope



class RHRAD_online:
    
    def __init__(self, 
                 hr="AHYIJDV_hr.csv", 
                 steps="AHYIJDV_steps.csv",
                 myphd_id="myphd_id",
                 symptom_date="NaN",
                 diagnosis_date="NaN",
                 RANDOM_SEED=1337,
                 outliers_fraction=0.1,
                 baseline_window=480, # 20 days
                 sliding_window=1,
                 myphd_id_anomalies="myphd_id_anomalies.csv",
                 myphd_id_figure1 = "myphd_id_anomalies.pdf",
                 myphd_id_alerts = "myphd_id_alerts.csv",
                 myphd_id_figure2 = "myphd_id_alerts.pdf",
                 last_day_only=True
                 ):
    
        # Initialize Variables
        self.fitbit_hr = hr
        self.fitbit_steps = steps
        self.myphd_id = hr.split("_")[0]
        self.symptom_date = symptom_date
        self.diagnosis_date = diagnosis_date
        self.RANDOM_SEED = RANDOM_SEED
        self.outliers_fraction =  outliers_fraction
        self.baseline_window = baseline_window #744
        self.sliding_window = sliding_window
        
        self.myphd_id_anomalies = self.myphd_id+"_anomalies.csv"
        self.myphd_id_figure1 = myphd_id_figure1
        self.myphd_id_alerts = self.myphd_id+"_alerts.csv"
        self.myphd_id_figure2 = myphd_id_figure2

        self.last_day_only = last_day_only
        
        # Process data
        df1 = self.resting_heart_rate(self.fitbit_hr, self.fitbit_steps) # RHR df at 1min resolution
        df2 = self.pre_processing(df1) # RHR df, smoothed and at 1hr resolution

        # Apply seasonality correction
        sdHR = df2[['heartrate']]
        data_seasnCorec = self.seasonality_correction(sdHR)
        data_seasnCorec += 0.1

        # Run model
        self.dfs = []
        self.data_train = []
        self.data_test = []

        self.online_anomaly_detection(data_seasnCorec, self.baseline_window, self.sliding_window)

        # print(self.data_test)

        # Process results
        self.results = self.merge_test_results(self.data_test)
        self.positive_anomalies = self.positive_anomalies(self.results)

        self.alerts = self.create_alerts(self.positive_anomalies, self.results, self.fitbit_hr)
        self.test_alerts = self.merge_alerts(self.results, self.alerts)



        
    # Infer resting heart rate ------------------------------------------------------
    def resting_heart_rate(self, heartrate, steps):
        """
        This function uses heart rate and steps data to infer resting heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
        """
        # heart rate data
        df_hr = pd.read_csv(heartrate)
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        # steps data
        df_steps = pd.read_csv(steps)
        df_steps = df_steps.set_index('datetime')
        df_steps.index.name = None
        df_steps.index = pd.to_datetime(df_steps.index)

        # merge heartrate and steps
        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        df1 = df1.resample('1min').mean() # resample to 1min resolution
        df1 = df1.dropna()
        
        # define RHR as the HR measurements recorded when there were zero steps taken during a rolling time window of the preceding 12 minutes (including the current minute)
        df1['steps_window_12'] = df1['steps'].rolling(12).sum()
        df1 = df1.loc[(df1['steps_window_12'] == 0)]
        return df1
    
    # Pre-processing ------------------------------------------------------
    def pre_processing(self, resting_heart_rate):
        """
        This function takes resting heart rate data and applies moving averages to smooth the data and 
        downsamples to one hour by taking the avegare values
        """
        df1 = resting_heart_rate

        # smooth data
        df_nonas = df1.dropna()
        df1_rom = df_nonas.rolling(400).mean()
        # resample
        df1_resmp = df1_rom.resample('1H').mean()
        df2 = df1_resmp.drop(['steps', 'steps_window_12'], axis=1)
        df2 = df2.dropna()
        return df2
    
    # Seasonality correction ------------------------------------------------------
    def seasonality_correction(self, resting_heart_rate):
        """
        This function takes output pre-processing and applies seasonality correction
        """
        sdHR = resting_heart_rate

        sdHR_decomposition = seasonal_decompose(sdHR, model='additive', freq=1)
        sdHR_decomp = pd.DataFrame(sdHR_decomposition.resid + sdHR_decomposition.trend)
        sdHR_decomp.rename(columns={sdHR_decomp.columns[0]:'heartrate'}, inplace=True)
    
        return sdHR_decomp
    
    # Train model and predict anomalies ------------------------------------------------------
    def online_anomaly_detection(self, data_seasnCorec, baseline_window, sliding_window):
        """
        data_seasnCorec comes from previous step
        baseline_window and sliding_window are both (int) types -- lengths of respective windows
        
        # split the data, standardize the data inside a sliding window 
        # parameters - 1 month baseline window and 1 hour sliding window
        # fit the model and predict the test set
        """
        if(self.last_day_only):
            data_train_w = data_seasnCorec[-1-baseline_window:-1] 
            # train data normalization ------------------------------------------------------
            data_train_w += 0.1
            standardizer = StandardScaler().fit(data_train_w.values)
            data_train_scaled = standardizer.transform(data_train_w.values)
            data_train_scaled_features = pd.DataFrame(data_train_scaled, index=data_train_w.index, columns=data_train_w.columns)
            
            data = pd.DataFrame(data_train_scaled_features)
            data_1 = pd.DataFrame(data).fillna(0)
            data_train_w = data_1
            self.data_train.append(data_train_w)

            data_test_w = data_seasnCorec[-1:] 
            # test data normalization ------------------------------------------------------
            data_test_w += 0.1
            data_test_scaled = standardizer.transform(data_test_w.values)
            data_scaled_features = pd.DataFrame(data_test_scaled, index=data_test_w.index, columns=data_test_w.columns)
            
            data = pd.DataFrame(data_scaled_features)
            data_1 = pd.DataFrame(data).fillna(0)
            data_test_w = data_1
            self.data_test.append(data_test_w)

            # fit the model  ------------------------------------------------------
            model = EllipticEnvelope(random_state=self.RANDOM_SEED,
                                    contamination=self.outliers_fraction,
                                    support_fraction=0.7).fit(data_train_w)
            # predict the test set
            preds = model.predict(data_test_w)
            #preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
            self.dfs.append(preds)
    
        else:
            for i in range(baseline_window, len(data_seasnCorec)):
                data_train_w = data_seasnCorec[i-baseline_window:i] 
                # train data normalization ------------------------------------------------------
                data_train_w += 0.1
                standardizer = StandardScaler().fit(data_train_w.values)
                data_train_scaled = standardizer.transform(data_train_w.values)
                data_train_scaled_features = pd.DataFrame(data_train_scaled, index=data_train_w.index, columns=data_train_w.columns)
                
                data = pd.DataFrame(data_train_scaled_features)
                data_1 = pd.DataFrame(data).fillna(0)
                data_train_w = data_1
                self.data_train.append(data_train_w)

                data_test_w = data_seasnCorec[i:i+sliding_window] 
                # test data normalization ------------------------------------------------------
                data_test_w += 0.1
                data_test_scaled = standardizer.transform(data_test_w.values)
                data_scaled_features = pd.DataFrame(data_test_scaled, index=data_test_w.index, columns=data_test_w.columns)
                
                data = pd.DataFrame(data_scaled_features)
                data_1 = pd.DataFrame(data).fillna(0)
                data_test_w = data_1
                self.data_test.append(data_test_w)

                # fit the model  ------------------------------------------------------
                model = EllipticEnvelope(random_state=self.RANDOM_SEED,
                                        contamination=self.outliers_fraction,
                                        support_fraction=0.7).fit(data_train_w)
                # predict the test set
                preds = model.predict(data_test_w)
                #preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
                self.dfs.append(preds)
    
    # Merge predictions ------------------------------------------------------
    def merge_test_results(self, data_test):
        """
        Merge predictions
        """
        # concat all test data (from sliding window) with their datetime index and others
        data_test = pd.concat(data_test)
        # merge predicted anomalies from test data with their corresponding index and other features 
        preds = pd.DataFrame(self.dfs)
        preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
        data_test_df = pd.DataFrame(data_test)
        data_test_df = data_test_df.reset_index()
        data_test_preds = data_test_df.join(preds)
        return data_test_preds
    
    # Positive Anomalies -----------------------------------------------------------------
    """
    Selects anomalies in positive direction and saves in a CSV file
    """
    def positive_anomalies(self, data):
        a = data.loc[data['anomaly'] == -1, ('index', 'heartrate')]
        positive_anomalies = a[(a['heartrate']> 0)]
        # Anomaly results
        positive_anomalies['Anomalies'] = self.myphd_id
        positive_anomalies.columns = ['datetime', 'std.rhr', 'name']
        positive_anomalies.to_csv(self.myphd_id_anomalies, header=True) 
        return positive_anomalies
    
    # Alerts  ------------------------------------------------------
    def create_alerts(self, anomalies, data, fitbit_oldProtocol_hr):
        """
        # creates alerts at every 24 hours and send at 9PM.
        # visualise alerts
        """
        # function to assign different alert names
        def alert_types(alert):
            if alert['alerts'] >=6:
                return 'RED'
            elif alert['alerts'] >=1:
                return 'YELLOW'
            else:
                return 'GREEN'

        # summarize hourly alerts
        anomalies = anomalies[['datetime']]
        anomalies['datetime'] = pd.to_datetime(anomalies['datetime'], errors='coerce')
        anomalies['alerts'] = 1
        anomalies = anomalies.set_index('datetime')
        anomalies = anomalies[~anomalies.index.duplicated(keep='first')]
        anomalies = anomalies.sort_index()
        alerts = anomalies.groupby(pd.Grouper(freq = '24H',  base=21)).cumsum()
        # apply alert_types function
        alerts['alert_type'] = alerts.apply(alert_types, axis=1)
        alerts_reset = alerts.reset_index()
        # save alerts
        #alerts.to_csv(myphd_id_alerts, mode='a', header=True) 


        # summarize hourly alerts to daily alerts
        daily_alerts = alerts_reset.resample('24H', on='datetime', base=21, label='right').count()
        daily_alerts = daily_alerts.drop(['datetime'], axis=1)

        # apply alert_types function
        daily_alerts['alert_type'] = daily_alerts.apply(alert_types, axis=1)


        # merge missing 'datetime' with 'alerts' as zero aka GREEN
        data1 = data[['index']]
        data1['alert_type'] = 0
        data1 = data1.rename(columns={"index": "datetime"})
        data1['datetime'] = pd.to_datetime(data1['datetime'], errors='coerce')
        data1 = data1.resample('24H', on='datetime', base=21, label='right').count()
        data1 = data1.drop(data1.columns[[0,1]], axis=1)
        data1 = data1.reset_index()
        data1['alert_type'] = 0

        data3 = pd.merge(data1, daily_alerts, on='datetime', how='outer')
        data4 = data3[['datetime', 'alert_type_y']]
        data4 = data4.rename(columns={ "alert_type_y": "alert_type"})
        daily_alerts = data4.fillna("GREEN")
        daily_alerts = daily_alerts.set_index('datetime')
        daily_alerts = daily_alerts.sort_index()


        # merge alerts with main data and pass 'NA' when there is a missing day instead of 'GREEN'
        df_hr = pd.read_csv(fitbit_oldProtocol_hr)

        df_hr['datetime'] = pd.to_datetime(df_hr['datetime'], errors='coerce')
        df_hr = df_hr.resample('24H', on='datetime', base=21, label='right').mean()
        df_hr = df_hr.reset_index()
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        df3 = pd.merge(df_hr, daily_alerts, how='outer', left_index=True, right_index=True)
        df3 = df3[df3.alert_type.notnull()]
        df3.loc[df3.heartrate.isna(), 'alert_type'] = pd.NA


        daily_alerts = df3.drop('heartrate', axis=1)
        daily_alerts = daily_alerts.reset_index()
        daily_alerts = daily_alerts.rename(columns={"index": "datetime"})
        daily_alerts.to_csv(self.myphd_id_alerts,  na_rep='NA', header=True) 

        return daily_alerts

    # Merge alerts  ------------------------------------------------------
    def merge_alerts(self, data_test, alerts):
        """
        Merge  alerts  with their corresponding index and other features 
        """

        data_test = data_test.reset_index()
        data_test['index'] = pd.to_datetime(data_test['index'], errors='coerce')
        test_alerts = alerts
        test_alerts = test_alerts.rename(columns={"datetime": "index"})
        test_alerts['index'] = pd.to_datetime(test_alerts['index'], errors='coerce')
        test_alerts = pd.merge(data_test, test_alerts, how='outer', on='index')
        test_alerts.fillna(0, inplace=True)

        return test_alerts

    
    # Visualization and save predictions ------------------------------------------------------
    def visualize(self, results, positive_anomalies, test_alerts, symptom_date, diagnosis_date):
        """
        visualize all the data with anomalies and alerts
        """
        try:

            with plt.style.context('fivethirtyeight'):

                fig, ax = plt.subplots(1, figsize=(80,15))
               
                ax.bar(test_alerts['index'], test_alerts['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                colors = {0:'', 'RED': 'red', 'YELLOW': 'yellow', 'GREEN': 'lightgreen'}
        
                for i in range(len(test_alerts)):
                    v = colors.get(test_alerts['alert_type'][i])
                    ax.vlines(test_alerts['index'][i], test_alerts['heartrate'].min(), test_alerts['heartrate'].max(),  linestyle='dotted',  lw=4, color=v)
                
                #ax.scatter(positive_anomalies['index'],positive_anomalies['heartrate'], color='red', label='Anomaly', s=500)

                ax.tick_params(axis='both', which='major', color='blue', labelsize=60)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=60)
                ax.set_title(myphd_id,fontweight="bold", size=50) # Title
                ax.set_ylabel('Std. RHR\n', fontsize = 50) # Y label
                ax.axvline(pd.to_datetime(symptom_date), color='grey', zorder=1, linestyle='--', marker="v" , markersize=22, lw=6) # Symptom date 
                ax.axvline(pd.to_datetime(diagnosis_date), color='purple',zorder=1, linestyle='--', marker="v" , markersize=22, lw=6) # Diagnosis date
                ax.tick_params(axis='both', which='major', labelsize=60)
                ax.tick_params(axis='both', which='minor', labelsize=60)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                ax.grid(zorder=0)
                ax.grid(True)
                plt.xticks(fontsize=30, rotation=90)
                plt.yticks(fontsize=50)
                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white')   
                figure = fig.savefig(myphd_id_figure1, bbox_inches='tight')                             
                return figure

        except:
            with plt.style.context('fivethirtyeight'):

                fig, ax = plt.subplots(1, figsize=(80,15))

                ax.bar(test_alerts['index'], test_alerts['heartrate'], linestyle='-', color='midnightblue', lw=6, width=0.01)

                colors = {0:'', 'RED': 'red', 'YELLOW': 'yellow', 'GREEN': 'lightgreen'}
        
                for i in range(len(test_alerts)):
                    v = colors.get(test_alerts['alert_type'][i])
                    ax.vlines(test_alerts['index'][i], test_alerts['heartrate'].min(), test_alerts['heartrate'].max(),  linestyle='dotted',  lw=4, color=v)
 
                #ax.scatter(positive_anomalies['index'],positive_anomalies['heartrate'], color='red', label='Anomaly', s=500)

                ax.tick_params(axis='both', which='major', color='blue', labelsize=60)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=60)
                ax.set_title(myphd_id,fontweight="bold", size=50) # Title
                ax.set_ylabel('Std. RHR\n', fontsize = 50) # Y label
                ax.tick_params(axis='both', which='major', labelsize=60)
                ax.tick_params(axis='both', which='minor', labelsize=60)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                ax.grid(zorder=0)
                ax.grid(True)
                plt.xticks(fontsize=30, rotation=90)
                plt.yticks(fontsize=50)
                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white')     
                figure = fig.savefig(myphd_id_figure1, bbox_inches='tight')       
                return figure






