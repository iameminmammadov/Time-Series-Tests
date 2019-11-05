import unittest
from unittest import TestCase
from pipelinetools.timeseries import Stationarity
import pandas as pd
import matplotlib.pyplot as plt
from dana_data_transform import DanaDataTransform


def plot_data(data, plot=False):
    missing = data.isnull().sum(0).reset_index()
    missing.columns = ['column', 'count']
    missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]
    missing['percentage'] = missing['count'] / float(data.shape[0]) * 100
    if plot is True:
        ind = np.arange(missing.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=(10,18))
        rects = ax.barh(ind, missing.percentage.values, color='r')
        ax.set_yticks(ind)
        ax.set_yticklabels(missing.column.values, rotation='horizontal')
        ax.set_xlabel("Precentage of missing values %", fontsize = 14)
        ax.set_title("Number of missing values in each column", fontsize = 18)
        plt.show()
    return missing

def load_data ():
    train_raw  = pd.read_pickle ('transformed-wideframe-April_Sep9.pkl')
    train_raw, dropped_raw = DanaDataTransform().generate_features(train_raw)
    train = train_raw.copy().sort_values('Timestamp_first')
    missing=plot_data(train, plot=False)
    delete_col = missing.loc[missing['percentage'] >= 20].column.values
    train = train.drop(delete_col, axis=1)
    
    for i in train.columns:
        train[i] = train[i].interpolate(method='linear')
    
    train.dropna(axis=0, inplace=True)
    
    train.set_index('Timestamp_first', inplace=True)
    
    cols_to_plot = [ 'OP_060_PALLET_LOAD_CARRIER_AND_COVER_CENTER_BORE_TO_SHIM_SURFACE_HOUSING',
                     'OP_060_PALLET_LOAD_CARRIER_AND_COVER_DIFF_LOWER_BEARING_DIAMETER_HOUSING',
                     'OP_060_PALLET_LOAD_CARRIER_AND_COVER_MATING_SURFACE_TO_PINION_CENTER_LINE_HOUSING',
                     'OP_060_PALLET_LOAD_CARRIER_AND_COVER_MATING_SURFACE_TO_SHIM_BEARING_SURFACE_COVER',
                     'OP_060_PALLET_LOAD_CARRIER_AND_COVER_MATING_SURFACE_TO_SHIM_SURFACE_HOUSING',
                     'OP_060_PALLET_LOAD_CARRIER_AND_COVER_PINION_HEAD_BEARING_DIAMETER_HOUSING',
                     'OP_130_GAUGE_PINION_BUTTON_POSITION',
                     'OP_130_GAUGE_PINION_HEAD_HEIGHT',
                     'OP_130_GAUGE_PINION_MOUNTING_POSITION_DEVIATION',
                     'OP_130_GAUGE_PINION_MOUNTING_POSITION',
                     'OP_180_BALANCE_ANGLE_FACE_RUNOUT',
                     'OP_180_BALANCE_ANGLE_FINAL_UNBALANCE',
                     'OP_180_BALANCE_ANGLE_HOLE_10',
                     'OP_180_BALANCE_ANGLE_HOLE_1',
                     'OP_180_BALANCE_ANGLE_HOLE_2',
                     'OP_180_BALANCE_ANGLE_HOLE_3',
                     'OP_180_BALANCE_ANGLE_HOLE_4',
                     'OP_180_BALANCE_ANGLE_HOLE_5',
                     'OP_180_BALANCE_ANGLE_HOLE_6',
                     'OP_180_BALANCE_ANGLE_HOLE_7',
                     'OP_180_BALANCE_ANGLE_HOLE_8',
                     'OP_180_BALANCE_ANGLE_HOLE_9',
                     'OP_180_BALANCE_ANGLE_INITIAL_UNBALANCE',
                     'OP_180_BALANCE_ANGLE_RADIAL_RUNOUT',
                     'OP_180_BALANCE_CORRECTION_PASSES',
                     'OP_180_BALANCE_DEPTH_HOLE_10',
                     'OP_180_BALANCE_DEPTH_HOLE_1',
                     'OP_180_BALANCE_DEPTH_HOLE_2',
                     'OP_180_BALANCE_DEPTH_HOLE_3',
                     'OP_180_BALANCE_DEPTH_HOLE_4',
                     'OP_180_BALANCE_DEPTH_HOLE_5',
                     'OP_180_BALANCE_DEPTH_HOLE_6',
                     'OP_180_BALANCE_DEPTH_HOLE_7',
                     'OP_180_BALANCE_DEPTH_HOLE_8',
                     'OP_180_BALANCE_DEPTH_HOLE_9',
                     'OP_180_BALANCE_FLANGE_INNER_BORE_DEVIATION',
                     'OP_180_BALANCE_MAGNITUDE_FACE_RUNOUT',
                     'OP_180_BALANCE_MAGNITUDE_FINAL_UNBALANCE',
                     'OP_180_BALANCE_MAGNITUDE_INITIAL_UNBALANCE',
                     'OP_180_BALANCE_MAGNITUDE_RADIAL_RUNOUT',
                     'OP_180_BALANCE_NUMBER_OF_HOLES',
                     'OP_180_BALANCE_SORT_KEY',
                     'OP_190_GAUGE_PINION_DRAG_TORQUE',
                     'OP_190_GAUGE_PINION_TORQUE_DRAG',
                     'OP_190_GAUGE_PINION_TORQUE_MAXIMUM',
                     'OP_190_GAUGE_PINION_TORQUE_MEAN',
                     'OP_190_GAUGE_PINION_TORQUE_MINIMUM',
                     'OP_230_PRESS_DIFF_CUP_ACTUAL_SHIM_COVER_SIDE',
                     'OP_230_PRESS_DIFF_CUP_ACTUAL_SHIM_HOUSING_SIDE',
                     'OP_230_PRESS_DIFF_CUP_BEARING_TORQUE_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_BOTTOM_OUT_FORCE_COVER_SIDE',
                     'OP_230_PRESS_DIFF_CUP_BOTTOM_OUT_FORCE_HOUSING_SIDE',
                     'OP_230_PRESS_DIFF_CUP_BOTTOM_OUT_POSITION_COVER_SIDE',
                     'OP_230_PRESS_DIFF_CUP_BOTTOM_OUT_POSITION_HOUSING_SIDE',
                     'OP_230_PRESS_DIFF_CUP_CUP_MID_STROKE_FORCE_COVER_SIDE_MIN',
                     'OP_230_PRESS_DIFF_CUP_CUP_MID_STROKE_FORCE_HOUSING_SIDE_MIN',
                     'OP_230_PRESS_DIFF_CUP_DIFF_CASSE_HEIGHT_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_DIFF_FLAT_TO_BOTTOM_CUP_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_DIFF_TOP_CUP_TO_FLAT_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_DIFF_TORQUE_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_MID_STROKE_FORCE_COVER_SIDE_MAX',
                     'OP_230_PRESS_DIFF_CUP_MID_STROKE_FORCE_HOUSING_SIDE_MAX',
                     'OP_230_PRESS_DIFF_CUP_PRELOAD_VGAGE',
                     'OP_230_PRESS_DIFF_CUP_PRODUCTION_OFFSET_COVER_SIDE',
                     'OP_230_PRESS_DIFF_CUP_PRODUCTION_OFFSET_HOUSING_SIDE',
                     'OP_230_PRESS_DIFF_CUP_SHIM_CALL_COVER_SIDE',
                     'OP_230_PRESS_DIFF_CUP_SHIM_CALL_HOUSING_SIDE',
                     'OP_230_PRESS_DIFF_CUP_SHIM_PART_OFFSET_COVER',
                     'OP_230_PRESS_DIFF_CUP_SHIM_PART_OFFSET_HOUSING',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_10',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_11',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_12',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_1',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_2',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_3',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_4',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_5',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_6',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_7',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_8',
                     'OP_270_RUNDOWN_COVER_BOLT_ANGLE_BOLT_9',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_10',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_11',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_12',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_1',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_2',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_3',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_4',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_5',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_6',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_7',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_8',
                     'OP_270_RUNDOWN_COVER_BOLT_TORQUE_BOLT_9',
                     'OP_280_GAUGE_BACKLASH_BACKLASH_MAX',
                     'OP_280_GAUGE_BACKLASH_BACKLASH_MEAN',
                     'OP_280_GAUGE_BACKLASH_BACKLASH_MIN',
                     'OP_280_GAUGE_BACKLASH_BACKLASH_SPAN',
                     'OP_280_GAUGE_BACKLASH_COLLET_SLIPPAGE',
                     'OP_280_GAUGE_BACKLASH_GEAR_RATIO',
                     'OP_280_GAUGE_BACKLASH_TORQUE_DELTA',
                     'OP_280_GAUGE_BACKLASH_TORQUE_DIFF',
                     'OP_280_GAUGE_BACKLASH_TORQUE_DRAG',
                     'OP_280_GAUGE_BACKLASH_TORQUE_PINION',
                     'OP_280_GAUGE_BACKLASH_TORQUE_TOTAL']
    
    cols = [col for col in train.columns if col in cols_to_plot]
    train_df = train[cols]
    
    return train_df


class TestStationarity(TestCase):
    def test_trend (self):
        train_df = load_data()
        tstest = Stationarity(granularity = 10) 
        # For Series
        trend_coeff = tstest.is_trend(train_df['OP_180_BALANCE_ANGLE_FINAL_UNBALANCE'])
        self.assertEqual (trend_coeff, 1)
        #For DF
        trend_coeff = tstest.is_trend(train_df)
        self.assertTrue (0<=trend_coeff<=2)
    
    def test_stationarity (self):
        train_df = load_data()
        tstest = Stationarity (granularity = 10)
        # For Series
        season_coeff = tstest.is_seasonality(train_df['OP_180_BALANCE_ANGLE_FINAL_UNBALANCE'])
        self.assertEqual (season_coeff, 1230)
        # For DF
        season_coeff = tstest.is_seasonality(train_df)
        for i in season_coeff:
            self.assertTrue (1008<=i<=4032)

if __name__ == '__main__':
    unittest.main()

    