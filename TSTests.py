class Stationarity:
    '''
    This class is used to test for stationarity of passed time series dataset. 
    The following tests are invoked from pdmarima library and used to test
    the time series for the trend:
        Kwiatkowski–Phillips–Schmidt–Shin (KPSS);
        Augmented Dickey Fuller (ADF);
        Phillips=Perron (PP).
    The return result of these tests will indicate the order of differencing, needed
    to make dataset trend-invariant. 
    
    To detect seasonality, power spectral density is used. Following the suggestion at
    [1], the overall trend is removed. The frequency that correspond to peaks indicate 
    seasonality. The return result of this test indicates periods that display 
    seasonality, and can be further used to remove seasonality factor.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        A DataFrame or Series containing consecutive data points (timesteps). 
        The data can be either 2D, or 1D.
    granularity : int
        Sampling granularity in minutes, i.e., 10 minutes, 5 minutes.
    freq: int (only for plot)
        Frequence of the series. Used to initial exploration of time series.

    Returns
    -------
        A period of seasonality for either series, or individual columns in a dataframe.

    [1]: https://stats.stackexchange.com/questions/16117/what-method-can-be-used-to-detect-seasonality-in-data
    
    
    
    TODO: Make passed dataset stationary. 
    A draft code to do that is:
    def make_stationary (self, data):
        trend_diff_factor = self.is_trend(data)
        if trend_diff_factor>0:
            data_trend_stationary = pd.DataFrame(utils.diff(data, lag = 1, differences = trend_diff_factor), columns=data.columns)
        else:
            data_trend_stationary = data
        data_trend_stationary.dropna(inplace=True)            
        
        season_diff_factor = self.is_seasonality(data_trend_stationary)
        if season_diff_factor>0:
            data_season_stationary = pd.DataFrame(utils.diff(data, lag =1, differences=self.granularity), columns=data.columns) #maybe data_trend_stationary
        else:
            data_season_stationary = data_trend_stationary
        data_season_stationary.dropna(inplace=True)   
        return data_season_stationary
    
    '''
    
    def __init__ (self,granularity):
        self.granularity = 1/(60*granularity)
    
    def is_seasonality (self, data):
        trend_diff_factor = self.is_trend(data)
        #train_df_angle.name
        if trend_diff_factor>0:
            if data.ndim == 1:
                data_trend_stationary = pd.Series(utils.diff(data, lag = 1, differences = trend_diff_factor), name=data.name)
            else:
                data_trend_stationary = pd.DataFrame(utils.diff(data, lag = 1, differences = trend_diff_factor), columns=data.columns)
        else:
            data_trend_stationary = data
        data_trend_stationary.dropna(inplace=True)  

        import warnings
        warnings.simplefilter("error", RuntimeWarning)
        
        spectral_density = pd.DataFrame()
        if data_trend_stationary.ndim == 1: 
            f, Pxx = periodogram (data_trend_stationary, fs=self.granularity)
            spectral_density['f'] = f
            spectral_density['spectral density'] = Pxx
            spectral_density.sort_values(by=['spectral density'], axis=0, ascending=False, inplace=True)
            seasonality_coeff = (1/(spectral_density.loc[spectral_density['spectral density']==max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
            return seasonality_coeff
        else:
            seasonality_coeff_df = list()
            for i in data.columns:
                f, Pxx = periodogram (data_trend_stationary[i], fs=0.00167)        
                spectral_density['f'] = f
                spectral_density['spectral density'] = Pxx
                spectral_density.sort_values(by=['spectral density'], axis=0, ascending=False, inplace=True)
                try:
                    seasonality_coeff = (1/(spectral_density.loc[spectral_density['spectral density']==max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                    seasonality_coeff_df.append(seasonality_coeff)
                except RuntimeWarning:
                    print ('Division by 0. Seasonality coefficient will not be calculated for this feature!')
                    pass
            return seasonality_coeff_df
                
    # It will return the maximum coefficient for trend differencing. That will need to be passed to make_stationary function
    @staticmethod
    def is_trend (data):
        #using .ndim to return dimension of dataframe/series:
        # 1 for one dimension (series); 2 for two dimension (dataframe)
        if data.ndim == 1:
            estim_d_kp = ndiffs(data, test='kpss', max_d=4)
            estim_d_pp = ndiffs(data, test='pp', max_d = 4)
            estim_d_adf = ndiffs(data, test='adf', max_d = 4)
            return (estim_d_kp+estim_d_pp+estim_d_adf)
        else:
            estim_d_kp_list = []
            estim_d_pp_list = []
            estim_d_adf_list = []
            for i in data.columns:
                estim_d_kp_list.append(ndiffs(data[i], test='kpss', max_d=4))
                estim_d_pp_list.append(ndiffs(data[i], test='pp', max_d = 4))
                estim_d_adf_list.append(ndiffs(data[i], test='adf', max_d = 4))
            return max(estim_d_kp_list+estim_d_pp_list+estim_d_adf_list)
            
    @staticmethod
    def plot (data, freq, type='additive'):
        if data.ndim == 1:
            results_decompose = seasonal_decompose(data, model=type, freq=freq)
            results_decompose.plot()
        else:
            plt.rcParams.update({'figure.max_open_warning': 0})
            for i in data.columns:
                results_decompose = seasonal_decompose(data[i], model=type, freq=freq)
                results_decompose.plot()
