from download_game_level_data import DataLoaderGame
from datetime import date, timedelta, datetime

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from exports import EmailHandler, SQLHandler

import dataframe_image as dfi
import pytz
import numpy as np
import pandas as pd
import functions_framework
import os

N_DIGITS = 2
def rmse(y,y_hat):return np.sqrt(np.mean((y-y_hat)**2))

def pretty_time(utc_dt):
    local_tz = pytz.timezone('US/Eastern')
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt).strftime('%m/%d %I.%M%p')

def train_xg_early_stop(data, no_validation=False):
    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict = data
    xg_early_stop = xgb.XGBRegressor(tree_method="hist", early_stopping_rounds=4, enable_categorical=True, max_depth=4,
                                     reg_lambda=1.2, reg_alpha=0.9,
                                     subsample=0.5)
    xg_early_stop.fit(x_train, y_train, eval_set=[(x_validate, y_validate)]);
    pred_future = np.round(xg_early_stop.predict(x_predict), N_DIGITS)
    e = np.nan
    if not no_validation:
        pred = xg_early_stop.predict(x_test)
        e = rmse(y_test, pred)
    return pred_future, e


def train_xg(data, no_validation=False):
    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict = data
    xg_full = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, max_depth=3, reg_lambda=5.3, reg_alpha=10.9,
                               subsample=0.8)
    xg_full.fit(np.r_[x_train, x_validate], np.r_[y_train, y_validate])
    pred_future = np.round(xg_full.predict(x_predict), N_DIGITS)
    e = np.nan
    if not no_validation:
        pred = xg_full.predict(x_test)
        e = rmse(y_test, pred)
    return pred_future, e


def train_lin_reg(data, no_validation=False):
    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict = data
    try:
        reg = LinearRegression().fit(np.r_[x_train, x_validate], np.r_[y_train, y_validate])
        pred_future = np.round(reg.predict(x_predict).flatten(), N_DIGITS)
        e = np.nan
        if not no_validation:
            pred = reg.predict(x_test).flatten()
            e = rmse(y_test, pred)
        return pred_future, e
    except:
        return np.zeros_like(x_test.shape[0]) * np.nan, np.nan


def create_images_and_send_emaiL(performance, predict_table):
    S = SQLHandler()
    S.write_new_table(performance, 'model_performance', False)  # dont overwrite -- lets accumulate bets...
    S.write_new_table(predict_table, 'predict_table', True)
    print('written to SQL DB')

    if not os.path.exists('images'): os.mkdir('images')
    dfi.export(performance, 'images/current_performance.png', table_conversion="matplotlib")
    pretty_table = predict_table.copy()
    pretty_table.date_time = pretty_table.date_time.apply(pretty_time)
    pretty_table = pretty_table.sort_values('game_id').reset_index(drop=True).drop(columns=['day'])
    pretty_table = pretty_table.style.background_gradient(subset=pretty_table.columns[-6:],
                                                          cmap="RdBu_r")
    dfi.export(pretty_table, 'images/current_predictions.png', table_conversion="matplotlib")  # quite slow

    E = EmailHandler()
    E.draft_msg(images=['images/current_performance.png', 'images/current_predictions.png'])
    E.send_msg()



def get_train_test_inds(data_use):
    now = datetime.now()
    today = date.today()
    tomorrow = today + timedelta(days=1)
    if now.hour > 19:  # if after 7PM, lets just predict for tomorrow
        predict_day_cutoff = tomorrow
    else:
        predict_day_cutoff = today
    two_week_ago = today - timedelta(days=14)
    four_week_ago = today - timedelta(days=28)

    t_start = f'{2021}-03-01'
    t_validate = four_week_ago.strftime('%Y-%m-%d')
    t_test = two_week_ago.strftime('%Y-%m-%d')

    ind_train = (data_use.date_time <= t_validate) & (data_use.date_time > t_start)
    ind_validate = (data_use.date_time > t_validate) & (data_use.date_time < t_test)
    ind_test = (data_use.date_time >= t_test) & (data_use.date_time < date.today().strftime('%Y-%m-%d'))

    return ind_train, ind_validate, ind_test

def get_target_data(test=False):
    dl = DataLoaderGame()
    if not test:
        dl.update_schedule()  #save time for now

    data = dl.import_merged_data()
    future_games = dl.import_upcoming()
    return data, future_games

def run_models(data_use_model, data_use_pred, ind_train, ind_validate, ind_test, no_validation=False):
    col_x = ['prev_pitching_k_game', 'prev_pitching_bb_game', 'prev_pitching_ip_game',
             'prev_hitting_BBp', 'prev_hitting_Kp',
             'pred_k_hitting', 'pred_bb_hitting',
             'pitching_k_inning', 'pitching_bb_inning']

    predict_table = data_use_pred.iloc[:, :8].copy()

    performance = []
    for col_y in ['k', 'bb']:

        x_train, y_train = data_use_model.loc[ind_train, col_x], data_use_model.loc[ind_train, col_y]
        x_validate, y_validate = data_use_model.loc[ind_validate, col_x], data_use_model.loc[ind_validate, col_y]
        x_test, y_test = data_use_model.loc[ind_test, col_x], data_use_model.loc[ind_test, col_y]
        x_predict = data_use_pred.loc[:, col_x]

        if no_validation:
            x_train, y_train = pd.concat([x_train, x_validate]), pd.concat([y_train, y_validate])
            x_validate, y_validate = x_test, y_test

        data = (x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict)
        # currently 3 models
        pred_xg_early, rmse_xg_early = train_xg_early_stop(data, no_validation)
        pred_xg, rmse_xg = train_xg(data, no_validation)
        pred_reg, rmse_reg = train_lin_reg(data, no_validation)

        performance.append({'xg_early': rmse_xg_early, 'xg': rmse_xg, 'reg': rmse_reg})
        predict_table[col_y + '_xg_early'] = pred_xg_early
        predict_table[col_y + '_xg'] = pred_xg
        predict_table[col_y + '_reg'] = pred_reg

    performance = pd.DataFrame(performance)
    return predict_table, performance

@functions_framework.http
def main(request, *args, **kwargs):
    return main_local(*args, **kwargs)

def main_local(test=False, use_imputed=True, no_validation=False,test_export=False):

    if test_export:
        performance = pd.read_parquet('performance.parquet')
        predict_table = pd.read_parquet('predict_table.parquet')
        create_images_and_send_emaiL(performance, predict_table)
        return 0

    data_use, future_games = get_target_data(test)

    todays_date_table = pd.DataFrame(columns=['date','time'],data=[[date.today(),datetime.now().strftime('%H:%M')]]) # add today date.
    S = SQLHandler()

    if len(future_games) == 0:

        S.write_new_table(todays_date_table, 'model_last_run', True)
        print('No Games Today')
        return 'NO GAMES'

    ind_train, ind_validate, ind_test = get_train_test_inds(data_use)
    print(f'train: {np.sum(ind_train)} validate: {np.sum(ind_validate)} test: {np.sum(ind_test)}')

    if use_imputed:
        impute_values = data_use.loc[ind_train].mean(numeric_only=True)
        data_use_model = data_use.copy().fillna(impute_values)
        data_use_pred = future_games.copy().fillna(impute_values)
    else:
        data_use_model = data_use
        data_use_pred = future_games

    predict_table, performance = run_models(data_use_model, data_use_pred, ind_train, ind_validate, ind_test,
                                            no_validation)

    # performance.to_parquet('performance.parquet')
    # predict_table.to_parquet('predict_table.parquet')
    # return performance, predict_table
    # create_images_and_send_emaiL(performance, predict_table)

    S.write_new_table(performance, 'model_performance', False)  # dont overwrite -- lets accumulate bets...
    S.write_new_table(predict_table, 'predict_table', True)
    S.write_new_table(todays_date_table, 'model_last_run', True)
    print('written to SQL DB')

    return 'success!'

if __name__ == '__main__':
    out = main_local()