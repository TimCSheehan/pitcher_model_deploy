#!/usr/bin/env python3
import os.path
from glob import glob
import pandas as pd
import numpy as np
from datetime import date
from packaging.version import Version
import random, pyarrow

assert Version(pyarrow.__version__) > Version("15.0.0"), 'Need to update pyarrow for safe import!'

try:
    from pybaseball import statcast
except ModuleNotFoundError:
    print('pybaseball not available -- downloading wont be possible')

def logit(p, eps=1e-4):
    if type(p) is np.ndarray:
        p[p < eps] = eps
        p[p > (1 - eps)] = (1 - eps)
    return np.log(p / (1 - p))


def I(x): return x


def shuffle_within_group(data, grouping_var, vars_want=None, dict_based=False):
    ''' NB: approximately O(N). Much faster than any built in pandas functionallity for this'''
    if dict_based:
        # by shuffling within group, only swap within group
        indici_dict = {}
        for i, hi in enumerate(data[grouping_var]):
            if hi in indici_dict:
                indici_dict[hi].append(i)
            else:
                indici_dict[hi] = [i]
        # shuffle each group...
        indicies_in_order = np.concatenate([this_list for this_list in indici_dict.values()])
        indicies_shuffled = np.concatenate([random.sample(this_list, len(this_list)) for this_list
                                            in indici_dict.values()])
        shuffled_inds = np.zeros_like(indicies_in_order)
        shuffled_inds[indicies_in_order] = indicies_shuffled
    else:
        list_of_shuffled_inds = []
        cur_group = data.loc[0, grouping_var]
        _this_list = []
        for i, hi in enumerate(data[grouping_var]):
            if hi == cur_group:
                _this_list.append(i)
            else:
                random.shuffle(_this_list)
                list_of_shuffled_inds.append(_this_list)
                cur_group = hi
                _this_list = [i]
        else:
            list_of_shuffled_inds.append(_this_list)  # final list...

        shuffled_inds = np.concatenate(list_of_shuffled_inds)
    if vars_want:
        return data.loc[shuffled_inds, vars_want].values
    else:
        return data.loc[shuffled_inds]

def get_data_root():
    # to work from both root and notebook folder
    base_root = './data/'
    if not os.path.exists(base_root):
        base_root = '.' + base_root
        assert os.path.exists(base_root), f'Data Path Not Found "{base_root}"'

    loc_statcast = base_root + '_statcast/'
    if not os.path.exists(loc_statcast):
        loc_statcast = base_root + 'statcast/'
        assert os.path.exists(loc_statcast), f'Statcast Path Not Found "{loc_statcast}"'
        print(f'Using public statcast location {loc_statcast}')
    else:
        print(f'Using private statcast location {loc_statcast}')
    return base_root, loc_statcast

class BaseballETL:
    """
    ETL class for pitch level data. Drops unnecessary columns, converts to categorical and float16 where appropriate.
    Groups data by month and year and saves to Parquet files.
    Helper functions to automatically process entire seasons.
    """

    def __init__(self, replace=False):
        self.data_root, self.loc_statcast = get_data_root()

        self.replace = replace
        self.end_dt = None
        self.start_dt = None
        self.grab_current_season = False

        self.loc_umpires = self.data_root + 'umpires/umpires.csv'  # from https://www.retrosheet.org/
        self.statcast_cols_want = ['pitch_type', 'description', 'stand', 'p_throws', 'home_team', 'away_team', 'type',
                                   'balls', 'strikes', 'plate_x', 'plate_z', 'outs_when_up', 'inning', 'inning_topbot',
                                   'game_pk', 'game_date',
                                   'release_pos_x', 'release_pos_y', 'release_speed', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
                                   'az',
                                   'sz_top', 'sz_bot', 'spin_axis',
                                   'batter', 'pitcher']

        self.to_float = ['plate_x', 'plate_z', 'release_pos_x', 'release_pos_y', 'release_speed', 'vx0', 'vy0', 'vz0',
                         'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'spin_axis']

        self.to_categorical = ['pitch_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type',
                               'description', 'inning_topbot', 'balls', 'strikes', 'outs_when_up', 'inning',
                               'batter', 'pitcher', 'game_pk', 'inning', 'outs_when_up', 'balls', 'strikes', 'Umpire']
        self._data_pitches = None
        self.data_pitches = None
        self._data_merged = None
        self.data_merged = None
        self.data_umpires = None

    def transform(self, shrink=True, merge_umpires=True, pull_missing_umpires=False):
        if merge_umpires:
            self.merge_pitches_umpires(pull_missing_umpires=pull_missing_umpires)

        # assign data to "data_merged" and subselect columns
        if self._data_merged is None:
            if self._data_pitches is None:
                raise ValueError('No data loaded!')
            self._data_merged = self._data_pitches
            self.data_merged = self._data_merged.loc[:, self.statcast_cols_want]
        else:
            self.data_merged = self._data_merged.loc[:, self.statcast_cols_want + ['Umpire']]

        if shrink:
            self.data_merged = self.shrink_pitches(self.data_merged)

    def extract_pitches(self):
        self._data_pitches = statcast(start_dt=self.start_dt, end_dt=self.end_dt)

    def extract_umpires(self):
        umpires = pd.read_csv(self.loc_umpires)
        umpires['Date'] = umpires['Date'].astype('datetime64[ns]')
        umpires['Umpire'] = umpires['Umpire'].astype('str')
        umpires['Home'] = umpires['Home'].astype('category')
        umpires['Away'] = umpires['Away'].astype('category')
        bad_acc = umpires.Acc.apply(lambda x: (x[-1] == '*') or (x == 'ND'))
        acc = np.zeros_like(umpires.Acc.values) * np.nan
        acc[~bad_acc] = umpires['Acc'].values[~bad_acc]
        umpires['Acc'] = acc.astype('float')
        self.data_umpires = umpires[['Date', 'Umpire', 'Home', 'Away', 'Acc']]

    def merge_pitches_umpires(self, pull_missing_umpires=False):
        if self.data_umpires is None:
            self.extract_umpires()

        merge_dat = pd.merge(self._data_pitches, self.data_umpires, left_on=['game_date', 'home_team'],
                             right_on=['Date', 'Home'], how='outer', indicator=True)
        merge_dat = merge_dat[merge_dat['_merge'] != 'right_only']

        # get missing umps - takes a minute
        if pull_missing_umpires:
            missing_umps_games = merge_dat.game_pk.unique()
            for game_pk in missing_umps_games:
                this_ump = self.get_ump_name_game_pk(game_pk)
                merge_dat.loc[merge_dat.game_pk == game_pk, 'Umpire'] = this_ump

        self._data_merged = merge_dat.drop(columns=['Home', 'Away', 'Acc', 'Date', '_merge'])

    def shrink_pitches(self, this_data):

        for cat in self.to_categorical:
            if cat in this_data:
                this_data[cat] = this_data[cat].astype('category')
            else:
                print('cat missing', cat)

        for fl in self.to_float:
            if fl in this_data:
                this_data[fl] = this_data[fl].astype(np.float16)
            else:
                print('float missing', fl)
        return this_data

    def extract_transform_date_range(self, start_dt='2017-06-24', end_dt='2017-06-26', shrink=True,
                                     merge_umpires=True, pull_missing_umpires=False):
        self.start_dt = start_dt
        self.end_dt = end_dt
        if self.replace or self.check_run():
            self.extract_pitches()
            self.transform(shrink=shrink, merge_umpires=merge_umpires, pull_missing_umpires=pull_missing_umpires)
        else:
            print('Data already saved. To override, set "replace" to True or "check_save" to false')

    def save_name(self):
        season, m0, d0 = self.start_dt.split('-')
        if self.grab_current_season:
            return f'{self.loc_statcast}{season}-ALL.parquet'
        _, m1, d1 = self.end_dt.split('-')
        return f'{self.loc_statcast}{season}-month-{m0}-{m1}.parquet'

    def check_run(self):
        if self.replace:
            return True
        elif os.path.exists(self.save_name()):
            return False
        else:
            return True

    def load_data(self):
        save_name = self.save_name()
        if self.check_run():
            self.data_merged.to_parquet(save_name)
            print(save_name, 'Saved!')
        else:
            print(save_name, 'already generated... skipping!')

    def ETL_season(self, season='2017', chunk='month', shrink=True, merge_umpires=True,
                   pull_missing_umpires=False):
        if chunk == 'month':
            month_ranges = (('03-25', '04-30'), ('05-01', '05-31'), ('06-01', '06-30'), ('07-01', '07-31'),
                            ('08-01', '08-31'), ('09-01', '10-01'))  # NB: start and end dates are INCLUSIVE!
            for month in month_ranges:
                self.extract_transform_date_range(start_dt=f'{season}-{month[0]}', end_dt=f'{season}-{month[1]}',
                                                  shrink=shrink, merge_umpires=merge_umpires,
                                                  pull_missing_umpires=pull_missing_umpires)
                self.load_data()
        elif chunk == 'current':
            self.grab_current_season = True
            today = date.today()
            assert str(today.year) == season, 'invalid, only valid for current season'
            month = ['3-25', date.today().strftime('%m-%d')]
            print(month)
            self.extract_transform_date_range(start_dt=f'{season}-{month[0]}', end_dt=f'{season}-{month[1]}',
                                              shrink=shrink, merge_umpires=merge_umpires,
                                              pull_missing_umpires=pull_missing_umpires)
            self.load_data()
            self.grab_current_season = False
        else:
            raise ValueError('Chunking method not setup yet. Must be: month')


class DataLoader(BaseballETL):
    """
    Subclass of BaseballETL for importing data into memory. Includes helper functions to add history markers
    extract + process columns for fitting specific models.
    """
    def __init__(self):
        super().__init__()
        self.data = None

    def import_data(self, season='*', fn=None, reshrink=True, add_history=False, **kwargs):
        if not fn:
            fn = self.loc_statcast + season + '*' + '.parquet'
        fls = glob(fn)
        D = []
        for fl in fls:
            _d = pd.read_parquet(fl)
            D.append(_d)
        self.data = pd.concat(D)[::-1].reset_index(drop=True)  # get in chronological order!

        # move umpire to first column
        if 'Umpire' in self.data:
            umps = self.data.pop('Umpire')
            self.data.insert(0, 'Umpire', umps)

        if reshrink:
            self.data = self.shrink_pitches(self.data)
        self.data['called_pitch'] = (self.data['description'] == 'called_strike') | (self.data['description'] == 'ball')
        self.data['strike'] = self.data['type'].values == 'S'
        self.data['season'] = self.data['game_date'].dt.year.astype('category')
        self.data['month'] = self.data['game_date'].dt.month.astype('category')
        if add_history:
            self.data = self.add_history_markers(self.data, **kwargs)
        return self.data

    @staticmethod
    def add_history_markers(data, do_shuffle=False, dict_based_shuffle=False):
        if do_shuffle:
            if do_shuffle == 'half_inning':
                data['half_inning'] = (data['game_pk'].astype(str) + '_' + data['inning'].astype(str) + data[
                    'inning_topbot'].astype(str)).astype(str)
            elif do_shuffle not in data:
                raise ValueError('Invalid shuffle variable!')
            print('SHUFFLING: ', do_shuffle)
            out = shuffle_within_group(data, do_shuffle, ['plate_x', 'plate_z'], dict_based=dict_based_shuffle)
            data[['prev_plate_x', 'prev_plate_z']] = out
        else:
            data[['prev_plate_x', 'prev_plate_z']] = data[['plate_x', 'plate_z']].shift().values
        data['d_plate_x'] = data['plate_x'] - data['prev_plate_x']
        data['d_plate_z'] = data['plate_z'] - data['prev_plate_z']
        data['d_euc'] = np.sqrt(data['d_plate_x'] ** 2 + data['d_plate_z'] ** 2)
        return data

    @staticmethod
    def one_hot(data, map_type, n_samples=None, outcomes=None):
        if n_samples is None:
            n_samples = len(data)

        if map_type == 'outcome':
            if outcomes is None:
                outcomes = ('ball', 'called_strike', 'swinging_strike')
            one_hot = np.zeros((len(outcomes), n_samples))

            for oi, o in enumerate(outcomes):
                if o == 'swinging_strike':
                    one_hot[oi] = ((data['type'].values == 'S') &
                                   (data['description'].values != 'called_strike'))[:n_samples]
                elif o == 'ball':
                    one_hot[oi] = (data['type'].values == 'B')[:n_samples]
                else:
                    one_hot[oi] = data['description'].values[:n_samples] == o

        elif map_type == 'count':
            n_feat = 4 * 3
            one_hot = np.zeros((n_feat, n_samples))
            i = 0
            outcomes = []
            for n_strike in range(3):
                for n_ball in range(4):
                    these_inds = ((data['balls'].values == n_ball) & (data['strikes'].values == n_strike))[
                                 :n_samples].astype(bool)
                    one_hot[i, these_inds] = 1
                    outcomes.append(f'count-{n_ball}-{n_strike}')
                    i += 1
        else:
            raise ValueError('Invalid map_type')

        return one_hot, outcomes

    @staticmethod
    def get_nb(x, nb, fill=0):
        ind = -nb
        if ind < 0:
            d_this = np.concatenate((np.ones(-ind) * fill, x[:ind]))
        elif ind > 0:
            d_this = np.concatenate((x[ind:], np.ones(ind) * fill))
        else:
            raise
        return d_this

    @staticmethod
    def get_pitch_params_default(n_sd=4, box_bot_top=False, batter_lr=False):
        if box_bot_top:
            walls = [-0.9, 0.9, 0.0, 0.0]
        else:
            walls = [-0.9, 0.9, 1.5, 3.7]
        if batter_lr:
            walls += walls[:2]
        sd = [-2.0] * n_sd
        return walls + sd

    @staticmethod
    def get_pitch_use(data, batter=None):
        # pitch_use = data['called_pitch'] & ~data.loc[:, 'plate_x'].isna() & ~data.loc[:, 'plate_z'].isna()
        pitch_use = data['called_pitch'] & ~data[['plate_x', 'plate_z']].isna().all(1)
        if batter:
            assert batter in ('L', 'R'), 'Invalid batter flag'
            pitch_use &= (data['stand'] == batter)
        return pitch_use

    @classmethod
    def get_pitch_data(cls, data, pitch_use=None, n=None, get_plate_x_hist=None, n_sd=4, sz_bt=False, batter_lr=False):
        if pitch_use is None:
            pitch_use = cls.get_pitch_use(data)
        p0 = cls.get_pitch_params_default(n_sd=n_sd, box_bot_top=sz_bt, batter_lr=batter_lr)
        pitches = data.loc[pitch_use, ['plate_x', 'plate_z']].astype('float').values
        strike = data.loc[pitch_use, 'type'].values == 'S'
        if n is None:
            n = len(strike)

        dat = [pitches[:n, 0], pitches[:n, 1]]
        var_names = ['pitch_x', 'pitch_y']

        # handle requests separately
        # - data order: pitch xy, sz_bot/top, batter, prev_pitch
        if sz_bt:
            sz_rng = data.loc[pitch_use, ['sz_bot', 'sz_top']].astype('float').values
            dat += [sz_rng[:n, 0], sz_rng[:n, 1]]
            var_names += ['sz_bot', 'sz_top']

        if batter_lr:
            batter_right = data.loc[pitch_use, 'stand'].values == 'R'
            dat.append(batter_right[:n])
            var_names.append('batter_right')

        if get_plate_x_hist:
            hist_x = data.loc[pitch_use, 'prev_plate_x'].astype('float').values
            hist_x[np.isnan(hist_x)] = 0
            dat.append(hist_x[:n])
            var_names.append('prev_pitch_x')
            p0 += [0]

        dat = tuple(dat)
        var_names = tuple(var_names)
        assert len(dat) == len(var_names), 'check that!'
        kwargs = {'n_sd': n_sd, 'sz_bt': sz_bt, 'batter_lr': batter_lr, 'mod_wall_relu': get_plate_x_hist}

        return dat, strike[:n].astype(np.int8), var_names, p0, kwargs

    @classmethod
    def generate_logistic_data(cls, data, pitch_use=None, n=None, nb_get=5, add_plus_1=True, training_want='stack',
                               add_intercept=True, count_numerical=False, add_p_strike=False, outcomes=None,
                               link_p_strike=logit, ):
        one_hot_outcomes, outcomes = cls.one_hot(data, map_type='outcome', outcomes=outcomes)
        one_hot_count, count_strs = cls.one_hot(data, map_type='count')

        nb_stack = np.arange(1, nb_get + 1)
        one_hot_list, outcome_names = [], []
        for nb in nb_stack:
            one_hot_shift = np.concatenate((one_hot_outcomes[:, nb:], one_hot_outcomes[:, -nb:]), 1)
            one_hot_list.append(one_hot_shift)
            [outcome_names.append(f'B{nb}-{out}') for out in outcomes]
        if add_plus_1:
            one_hot_list.append(np.concatenate((np.zeros((one_hot_outcomes.shape[0], 1)),
                                                one_hot_outcomes[:, :-1]), 1))
            [outcome_names.append(f'F1-{out}') for out in outcomes]

        one_hot_stack = np.vstack(one_hot_list)
        if pitch_use is None:
            pitch_use = cls.get_pitch_use(data)
        if n is None:
            n = np.sum(pitch_use)
        strike = data.loc[pitch_use, 'type'].values[:n] == 'S'
        outcomes_use = one_hot_stack[:, pitch_use][:, :n]
        if count_numerical:
            count_use = data.loc[pitch_use, ['balls', 'strikes']].values.astype(int)[:n].T
            count_strs = ['balls', 'strikes']
        else:
            count_use = one_hot_count[:, pitch_use][:, :n]

        n_samp = len(strike)
        intercept = np.ones((n_samp, 1 * add_intercept))  # so no dim if no intercept
        var_intercept = ['1'][:add_intercept]

        if add_p_strike:
            p_strike = link_p_strike(data.loc[pitch_use, ['pStrike']].values[:n])
        else:
            p_strike = np.zeros((n_samp, 0))
        var_p_strike = ['strike_zone'][:add_p_strike]

        if training_want is None:
            return outcomes_use, count_use, strike
        elif training_want == 'outcome':
            dat0 = outcomes_use.T
            var_names = outcome_names
        elif training_want == 'count':
            dat0 = count_use.T
            var_names = count_strs
        elif training_want == 'stack':
            dat0 = np.r_[outcomes_use, count_use].T
            var_names = outcome_names + count_strs
        else:
            raise ValueError('not a valid training_want value')

        var_names = var_intercept + var_names + var_p_strike
        dat0_full = np.c_[intercept, dat0, p_strike]
        # dat = (dat0_full, strike)

        assert dat0_full.shape[1] == len(var_names), 'variable names incorrect!'
        p0 = [0] * len(var_names)
        return dat0_full, strike.astype(np.int8), var_names, p0

    @classmethod
    def get_joint_data(cls, data, batter=None, n=None, n_sd=4, get_plate_x_hist=False, sz_bt=False, batter_lr=False,
                       **args_logistic):
        pitch_use = cls.get_pitch_use(data, batter=batter)
        dat_pitch, _strike, var_names_pitch, p_sz, kwargs = cls.get_pitch_data(data, pitch_use=pitch_use, n=n,
                                                                               n_sd=n_sd,
                                                                               get_plate_x_hist=get_plate_x_hist,
                                                                               sz_bt=sz_bt, batter_lr=batter_lr)
        dat_logistic, strike, var_names_logistic, p_logistic = cls.generate_logistic_data(data, pitch_use=pitch_use,
                                                                                          n=n, **args_logistic)

        assert all(_strike == strike), 'strikes do not match up!'
        dat_out = (dat_pitch, dat_logistic)
        var_names_out = var_names_logistic

        p0 = p_sz + p_logistic
        return dat_out, strike.astype(np.int8), var_names_out, p0, kwargs