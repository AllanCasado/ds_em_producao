import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann:
    
    def __init__(self):
        self.home_path = '../' #'C:/Users/allan/Documents/GitHub/ds_em_producao/'
        self.scaler_competition_distance = pickle.load(open(self.home_path + "artifacts/scaler_competition_distance.pkl", "rb"))
        self.scaler_competition_time_month = pickle.load(open(self.home_path + "artifacts/scaler_competition_time_month.pkl", "rb"))
        self.scaler_promo_time_week = pickle.load(open(self.home_path + "artifacts/scaler_promo_time_week.pkl", "rb"))
        self.scaler_year = pickle.load(open(self.home_path + "artifacts/scaler_year.pkl", "rb"))
        self.encoder_state_holiday = pickle.load(open(self.home_path + "artifacts/encoder_state_holiday.pkl", "rb"))
        self.encoder_store_type = pickle.load(open(self.home_path + "artifacts/encoder_store_type.pkl", "rb"))
        
    
    def data_cleaning(self, df1):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 
            'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        #função para alterar o nome das colunas para o formato snakecase
        change_to_snakecase = lambda x: inflection.underscore(x)

        #aplicando a função nas colunas antigas
        new_cols = list(map(change_to_snakecase, cols_old))

        #renomeando as colunas
        df1.columns = new_cols
        
        #alterando o tipo da coluna date para datetime (estava como object)
        df1['date'] = pd.to_datetime(df1['date'])
        
        #substituindo dados faltantes
        max_value = df1['competition_distance'].max()
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: max_value*3 if math.isnan(x) else x)
        
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)
        
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)
        
        #diicionário para mapear os meses
        month_map = {
            1:'Jan',
            2:'Feb',
            3:'Mar',
            4:'Apr',
            5:'May',
            6:'Jun',
            7:'Jul',
            8:'Aug',
            9:'Sept',
            10:'Oct',
            11:'Nov',
            12:'Dec'
        }

        #preenche os nulos com 0
        df1['promo_interval'].fillna(0, inplace=True)

        #cria uma coluna 'month_map' com o nome dos meses
        df1['month_map'] = df1.date.dt.month.map(month_map)

        #cria uma coluna 'is_promo' que indica se no mês do registro teve Promo2 ou não
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
        
        # competition_open_since_month
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)

        # competition_open_since_year 
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        # promo2_since_week           
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)

        # promo2_since_year           
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    
    def feature_engineering(self, df2):
        
        #year
        df2['year'] = df2['date'].dt.year

        #month
        df2['month'] = df2['date'].dt.month

        #day
        df2['day'] = df2['date'].dt.day

        #week_of_year -> semana do ano
        df2['week_of_year'] = df2['date'].dt.strftime('%W')

        #year_week -> ano-semana do ano
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        #competition_since (formatado o dia que a competição começou no mesmo formato que 'date')
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)

        #competition_time_month (tempo em meses desde que a competição começou)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        #promo_since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w'))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)
        df2['promo_time_week'] = df2['promo_time_week'].apply(lambda x: 0 if x == -1 else x)

        #assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        #state_holiday a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public holiday' if x == 'a' else 'easter holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')
        
        #filtragem de variáveis
        df2 = df2[(df2['open'] != 0)] 
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    
    def data_preparation(self, df5):
        #rescaling
        df5['competition_distance'] = self.scaler_competition_distance.transform(df5[['competition_distance']].values)
        df5['competition_time_month'] = self.scaler_competition_time_month.transform(df5[['competition_time_month']].values)
        df5['promo_time_week'] = self.scaler_promo_time_week.transform(df5[['promo_time_week']].values)
        df5['year'] = self.scaler_year.transform(df5[['year']].values)

        #one hot encoding
        ohe_encoded_test_columns = self.encoder_state_holiday.transform(df5[['state_holiday']])
        ohe_encoded_test_dataframe = pd.DataFrame(ohe_encoded_test_columns, columns=self.encoder_state_holiday.get_feature_names_out())
        ohe_encoded_test_dataframe.reset_index(drop=True, inplace=True)
        df5.reset_index(drop=True, inplace=True)
        df5 = pd.concat([df5, ohe_encoded_test_dataframe], axis=1)
        df5 = df5.drop('state_holiday', axis=1)
        
        #label encoding
        df5['store_type'] = self.encoder_store_type.transform(df5[['store_type']])
        
        #ordinal encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)
        
        #nature transformation
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))
        df5['week_of_year_sin'] = df5['week_of_year'].astype(int).apply(lambda x: np.sin(x*(2*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].astype(int).apply(lambda x: np.cos(x*(2*np.pi/52)))
        
        return df5
    
    
    def feature_selection(self, df5):
        mask = ['promo', 'school_holiday', 'store_type', 'assortment', 'promo2',
       'promo2_since_week', 'promo2_since_year', 'year', 'promo_time_week',
       'state_holiday_christmas', 'state_holiday_easter holiday',
       'state_holiday_public holiday', 'state_holiday_regular_day',
       'month_cos', 'day_cos', 'day_of_week_sin', 'day_of_week_cos',
       'week_of_year_cos', 'store']
        
        return df5[mask]
    
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        
        #join prediction into orignial data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient="records", date_format="iso")