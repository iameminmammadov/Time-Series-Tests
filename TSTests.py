class StationarityCheck:
    '''This class is used to test for stationarity of passed time series dataset.
    The following tests are invoked from pdmarima library and used to test
    the time series for the trend:
        Kwiatkowski–Phillips–Schmidt–Shin (KPSS);
        Augmented Dickey Fuller (ADF);
        Phillips=Perron (PP).
    The return result of these tests will indicate the order of differencing,
    needed to make dataset trend-invariant.

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
    freq : int (only for plot)
        Frequence of the series. Used to initial exploration of time series.

    Returns
    -------
        A period of seasonality for either series, or individual columns in a dataframe.
    
    Note
    ----
    [1]: https://stats.stackexchange.com/questions/16117/what-method-can-be-used-to-detect-seasonality-in-data

    TODO: Make passed dataset stationary.
    A draft code to do that is:
    def make_stationary (self, data):
        trend_diff_factor = self.check_trend(data)
        if trend_diff_factor>0:
            data_trend_stationary = pd.DataFrame(utils.diff(data, lag = 1, differences = trend_diff_factor),
                                                columns=data.columns)
        else:
            data_trend_stationary = data
        data_trend_stationary.dropna(inplace=True)

        season_diff_factor = self.is_seasonality(data_trend_stationary)
        if season_diff_factor>0:
            data_season_stationary = pd.DataFrame(utils.diff(data, lag =1, differences=self.granularity),
                                                    columns=data.columns) #maybe data_trend_stationary
        else:
            data_season_stationary = data_trend_stationary
        data_season_stationary.dropna(inplace=True)
        return data_season_stationary

    TODO: If signal does not display trend, then that signal is taken as it is and passed to seasonality detection.
    In that case, the sampling period becomes extremely large. That needs to be further researched.

    '''
    warnings.simplefilter("error", RuntimeWarning)

    def __init__(self, granularity):
        self.granularity = 1 / (60 * granularity)

    def check_seasonality(self, data):
        if data.ndim == 1:
            trend_diff_factor = self.check_trend(data)
            if trend_diff_factor != 0:
                spectral_density = pd.DataFrame()
                data_trend_stationary = pd.Series(utils.diff(
                    data, lag=1, differences=trend_diff_factor), name=data.name)
                f, Pxx = periodogram(data_trend_stationary,
                                     fs=self.granularity)
                spectral_density['f'] = f
                spectral_density['spectral density'] = Pxx
                spectral_density.sort_values(by=['spectral density'],
                                             axis=0,
                                             ascending=False,
                                             inplace=True)
                seasonality_coeff = (1 / (spectral_density.loc[
                    spectral_density['spectral density'] == max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                try:
                    seasonality_coeff = (
                        1 / (spectral_density.
                             loc[spectral_density['spectral density'] == max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                except RuntimeWarning:
                    return (np.NAN)
                return seasonality_coeff
            else:
                spectral_density = pd.DataFrame()
                data_trend_stationary = data
                f, Pxx = periodogram(data_trend_stationary,
                                     fs=self.granularity)
                spectral_density['f'] = f
                spectral_density['spectral density'] = Pxx
                spectral_density.sort_values(by=['spectral density'],
                                             axis=0,
                                             ascending=False,
                                             inplace=True)
                try:
                    seasonality_coeff = (
                        1 / (spectral_density.
                             loc[spectral_density['spectral density'] == max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                except RuntimeWarning:
                    return (np.NAN)
                return seasonality_coeff
        else:  # DataFrame
            trend_diff_factor = self.check_trend(data)
            seasonality_coeff_df = list()
            temp_0 = pd.DataFrame()
            temp_1 = pd.DataFrame()
            for row, val in enumerate(trend_diff_factor['max_trend']):
                if val == 0:
                    spectral_density = pd.DataFrame()
                    col_name = trend_diff_factor['index'][row]
                    col_holder = pd.Series(data[col_name].values,
                                           name=col_name)
                    temp_0[col_name] = col_holder
                    f, Pxx = periodogram(temp_0[col_name].values,
                                         fs=self.granularity)
                    spectral_density['f'] = f
                    spectral_density['spectral density'] = Pxx
                    spectral_density.sort_values(by=['spectral density'],
                                                 axis=0,
                                                 ascending=False,
                                                 inplace=True)
                    try:
                        seasonality_coeff = (
                            1 / (spectral_density.
                                 loc[spectral_density['spectral density'] == max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                        seasonality_coeff_df.append(seasonality_coeff)
                    except RuntimeWarning:
                        seasonality_coeff = np.nan
                        seasonality_coeff_df.append(seasonality_coeff)
                else:
                    spectral_density = pd.DataFrame()
                    col_name = trend_diff_factor['index'][row]
                    col_holder = pd.Series(utils.diff(data[col_name],
                                                      lag=1,
                                                      differences=val),
                                           name=col_name)
                    temp_1[col_name] = col_holder
                    f, Pxx = periodogram(temp_1[col_name].values,
                                         fs=self.granularity)
                    spectral_density['f'] = f
                    spectral_density['spectral density'] = Pxx
                    spectral_density.sort_values(by=['spectral density'],
                                                 axis=0,
                                                 ascending=False,
                                                 inplace=True)
                    try:
                        seasonality_coeff = (
                            1 / (spectral_density.
                                 loc[spectral_density['spectral density'] == max(spectral_density['spectral density']), 'f'].iloc[0])).astype(int)
                        seasonality_coeff_df.append(seasonality_coeff)
                    except RuntimeWarning:
                        seasonality_coeff = np.nan
                        seasonality_coeff_df.append(seasonality_coeff)
            dataframe_seasonality_return = pd.DataFrame(
                seasonality_coeff_df, columns=['seasonality period'])
            dataframe_seasonality_return['Column name'] = data.columns
            return dataframe_seasonality_return

    @staticmethod
    def check_trend(data):
        if data.ndim == 1:
            estim_d_kp = ndiffs(data, test='kpss', max_d=4)
            estim_d_pp = ndiffs(data, test='pp', max_d=4)
            estim_d_adf = ndiffs(data, test='adf', max_d=4)
            return (max(*[estim_d_kp, estim_d_pp, estim_d_adf]))
        else:
            trend_dataframe = pd.DataFrame(index=data.columns,
                                           columns=['trend coeff'])
            trend_dataframe.reset_index(inplace=True)
            estim_d_kp_list = []
            estim_d_pp_list = []
            estim_d_adf_list = []
            for i in data.columns:
                estim_d_kp_list.append(ndiffs(data[i], test='kpss', max_d=4))
                estim_d_pp_list.append(ndiffs(data[i], test='pp', max_d=4))
                estim_d_adf_list.append(ndiffs(data[i], test='adf', max_d=4))
            trend_dataframe['trend_coeff_adf'] = estim_d_adf_list
            trend_dataframe['trend_coeff_pp'] = estim_d_pp_list
            trend_dataframe['trend_coeff_kp'] = estim_d_kp_list
            trend_dataframe['max_trend'] = trend_dataframe.max(axis=1)
            return (trend_dataframe[['index', 'max_trend']])

    @staticmethod
    def plot(data, freq, type='additive'):
        if data.ndim == 1:
            results_decompose = seasonal_decompose(data, model=type, freq=freq)
            results_decompose.plot()
        else:
            for i in data.columns:
                results_decompose = seasonal_decompose(data[i],
                                                       model=type,
                                                       freq=freq)
                results_decompose.plot()