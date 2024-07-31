from functools import cache
import statsapi, os.path
import pandas as pd
from datetime import date, timedelta, datetime
from dateutil import parser
import numpy as np
from pybaseball import pitching_stats_range, team_batting
from pybaseball import league_pitching_stats, playerid_reverse_lookup
from glob import glob

team_dict_map = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': "BOS",
    'Chicago Cubs': "CHC",
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': "CIN",
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': "LAA",
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'San Francisco Giants': 'SFG',
    'Seattle Mariners': "SEA",
    'St. Louis Cardinals': "STL",
    'Tampa Bay Rays': "TBR",
    'Texas Rangers': "TEX",
    'Toronto Blue Jays': "TOR",
    'Washington Nationals': "WSN"}

## hardcoded estimated # of at bats (from 2023 data)
xx = np.arange(9)
b=2.6
m=-.11
pred_num_at_bat = b+m*xx

def get_data_root():
    # to work from both root and notebook folder
    base_root = './data/'
    if ~os.path.exists(base_root):
        base_root = '.' + base_root
        assert os.path.exists(base_root), f'Data Path Not Found "{base_root}"'
    return base_root

class ScheduleETL:
    def __init__(self, replace=False):
        self.loc_schedule = get_data_root() + 'schedule_v1'
        self.loc_merged_schedule = get_data_root() + 'merged_schedule_v1'
        self.loc_upcoming = get_data_root() + 'upcoming_games.parquet'
        self.team_dict_map = team_dict_map
        self.replace = replace
        self.schedule = None
        self.schedule_unrolled = None
        self.batting_season = None
        self.pitching_season = None
        self.merged_schedule = None
        self.merged_schedule_shrunk = None

        self.upcoming_games = None

        self.col_to_cat = ['pitching_team', 'hitting_team','pitcher', 'home_away', 'umpire',]

    def update_schedule(self, buffer_rerun=2):
        today = date.today()
        tomorrows = today + timedelta(days=1)
        season = today.year

        old_schedule = pd.read_parquet(self.save_name_schedule(season))
        old_schedule['date_time'] = old_schedule.game_datetime.apply(parser.parse).dt.date

        first_day_run = np.minimum(old_schedule.date_time.max(), today) - timedelta(days=buffer_rerun)
        print(f'grabbing from {first_day_run.strftime("%Y-%m-%d")} to {tomorrows.strftime("%Y-%m-%d")}')
        dates_grab = list(self.daterange(first_day_run, tomorrows))
        print(dates_grab)

        new_schedule = self.get_outcomes_from_game_dates(dates_grab)
        self.schedule = new_schedule.query('status=="Final"').copy()
        self.shrink_schedule()
        old_schedule_use = old_schedule[old_schedule.date_time <= first_day_run].drop(columns=['date_time'])
        self.schedule = pd.concat([old_schedule_use, self.schedule])
        self.shrink_schedule()

        print(f'Saving', end='...')
        self.schedule.to_parquet(self.save_name_schedule(season))
        print(f'Saved to {self.save_name_schedule(season)}!')
        self.etl_merge_previous_season_stats(season,True) # save updated schedule

        # yesterday_ts = pd.to_datetime('today').floor('D') + timedelta(days=-1)
        upcoming_games = new_schedule.query('status != "Final" ')  # & date_time.dt.tz_convert(None) >=@yesterday_ts'

        if len(upcoming_games) == 0:
            # no games today
            #- SAVE EMPTY TABLE
            self.upcoming_games = upcoming_games
            self.upcoming_games.to_parquet(self.loc_upcoming)
            return

        upcoming_games = self.unroll_schedule(new_schedule.query('status != "Final"').copy(),
                                              allow_missing=True).reset_index(drop=True)

        upcoming_games_filled = self.impute_missing_box_scores(upcoming_games.copy())
        self.upcoming_games = self.shrink_merged_schedule(upcoming_games_filled,['box_imputed'])

        # save out
        self.upcoming_games.to_parquet(self.loc_upcoming)

    def impute_missing_box_scores(self, these_games):
        assert type(self.merged_schedule_shrunk is not None), 'Need to establish merged table to pull from!'

        vars_pitching_box = ['pitching_strikeOuts', 'pitching_baseOnBalls', 'pitching_atBats',
                             'pitching_inningsPitched']
        vars_hitting_box = ['hitting_strikeOuts', 'hitting_baseOnBalls', 'hitting_atBats']

        vars_pitching_prev = ['prev_pitching_IP', 'prev_pitching_k_game',
                              'prev_pitching_bb_game', 'prev_pitching_ip_game']
        vars_hitting_prev = ['pred_k_hitting', 'pred_bb_hitting', 'prev_hitting_Kp', 'prev_hitting_BBp']

        for v in vars_pitching_prev:
            these_games[v] = np.nan
        for v in vars_hitting_prev:
            these_games[v] = np.nan

        these_games['box_imputed'] = False
        for i, row in these_games.iterrows():
            # add prev-season insertion here
            this_pitcher = self.merged_schedule_shrunk.query('pitcher==@row.pitcher')
            if len(this_pitcher) > 0:
                pitcher_d = this_pitcher.iloc[-1]
                for v in vars_pitching_prev:
                    if v == 'pitching_bb_inning':
                        print(pitcher_d[v])
                    row[v] = pitcher_d[v]

            this_hitting_team = self.merged_schedule_shrunk.query('hitting_team==@row.hitting_team')
            if len(this_hitting_team) > 0:
                hitting_d = this_hitting_team.iloc[-1]
                for v in vars_hitting_prev:
                    row[v] = hitting_d[v]

            if row.k == -999:  # no box score
                row['box_imputed'] = True
                this_pitcher = self.schedule_unrolled.query('pitcher==@row.pitcher')
                if len(this_pitcher) > 0:
                    pitcher_d = this_pitcher.iloc[-1]
                    for v in vars_pitching_box:
                        row[v] = pitcher_d[v]

                this_hitting_team = self.schedule_unrolled.query('hitting_team==@row.hitting_team')
                if len(this_hitting_team) > 0:
                    hitting_d = this_hitting_team.iloc[-1]
                    for v in vars_hitting_box:
                        row[v] = hitting_d[v]

            these_games.loc[i] = row

        these_games_1 = self.extract_schedule_stats(these_games)
        return these_games_1


    def etl_outcomes_season(self, season: int, do_shrink: bool = True) -> None:
        if (not self.replace) and os.path.exists(self.save_name_schedule(season)):
            print(f'{self.save_name_schedule(season)} already generated!')
        else:
            dates_grab = list(self.daterange(date(season, 3, 25), date(season, 10, 1)))
            if any(np.array(dates_grab) > date.today()):
                dates_grab = list(np.array(dates_grab)[np.array(dates_grab) <= date.today()])
            print(f'Grabbing schedule for {season}', end='...')
            self.schedule = self.get_outcomes_from_game_dates(dates_grab)
            if do_shrink:
                self.shrink_schedule()
            print(f'Saving', end='...')
            self.schedule.to_parquet(self.save_name_schedule(season))
            print(f'Saved to {self.save_name_schedule(season)}!')

    def etl_merge_previous_season_stats(self, season: int, load_table: bool=False):
        if (self.schedule is not None) and (int(self.schedule.iloc[0,1][:4]) == season):
            print('Schedule in memory')
            pass
        elif os.path.exists(self.save_name_schedule(season)):
            print('loading schedule from hard-drive')
            self.schedule = pd.read_parquet(self.save_name_schedule(season))
        else:
            raise ValueError('schedule not created yet -best to do so explicitly by running etl_outcomes_season')

        print('Unrolling and extracting schedule stats.')
        self.schedule_unrolled = self.unroll_schedule(self.schedule)
        self.schedule_unrolled = self.extract_schedule_stats(self.schedule_unrolled)

        print('Extracting prev season batting and hitting stats.')
        self.extract_batting_stats_season(season-1)
        self.extract_pitching_stats_season(season-1)

        # merge
        merge_0 = self.schedule_unrolled.merge(
            self.batting_season, on='hitting_team', how='outer', sort=False)

        self.merged_schedule = merge_0.merge(
            self.pitching_season, left_on='pitcher', right_on='full_name', how='left', sort=False)

        # ditch any rows where game_pk is nan
        bad_rows = self.merged_schedule.game_id.isna()
        if np.any(bad_rows):
            print(f'Removing {np.sum(bad_rows)} bad rows...')
            self.merged_schedule = self.merged_schedule.loc[~bad_rows]
        self.merged_schedule_shrunk = self.shrink_merged_schedule(self.merged_schedule)
        if load_table:
            self.merged_schedule_shrunk.to_parquet(self.save_name_merged_schedule(season))

    def extract_batting_stats_season(self, season: int) -> None:
        batting_season = team_batting(season, league='all', ind=1)
        col_grab = ['hitting_team', 'BB%', 'K%']  # 'BB','SO','AB'
        inv_map = {v: k for k, v in self.team_dict_map.items()}
        batting_season['hitting_team'] = batting_season.Team.map(inv_map)
        self.batting_season = (batting_season[col_grab].
                               rename(columns={'BB%': 'prev_hitting_BBp', 'K%': 'prev_hitting_Kp'}))

    def extract_pitching_stats_season(self, season: int) -> None:
        pitching_prev_season = pitching_stats_range(f"{season}-03-25",
                                                    f"{season}-10-01")
        pitching_prev_season['mlbID_int'] = pitching_prev_season.mlbID.values.astype(int)
        names = playerid_reverse_lookup(pitching_prev_season['mlbID_int'].values.astype(int))
        names['full_name'] = [f'{f.name_first.title()} {f.name_last.title()}' for _, f in names.iterrows()]
        pitching_prev_season_u = pitching_prev_season[['mlbID_int', 'G', 'IP', 'ER', 'BB', 'SO']].merge(
            names[['full_name', 'key_mlbam']], how='outer', left_on='mlbID_int', right_on='key_mlbam')
        pitching_prev_season_u = pitching_prev_season_u.drop(columns='mlbID_int')
        pitching_prev_season_u['k_game'] = pitching_prev_season_u['SO'] / pitching_prev_season_u['G']
        pitching_prev_season_u['bb_game'] = pitching_prev_season_u['BB'] / pitching_prev_season_u['G']
        pitching_prev_season_u['ip_game'] = pitching_prev_season_u['IP'] / pitching_prev_season_u['G']

        self.pitching_season = pitching_prev_season_u.rename(
            columns={'k_game': 'prev_pitching_k_game', 'bb_game': 'prev_pitching_bb_game',
                     'ip_game': 'prev_pitching_ip_game', 'IP': 'prev_pitching_IP'}
        ).loc[:, [
                   'full_name', 'prev_pitching_k_game', 'prev_pitching_bb_game',
                   'prev_pitching_ip_game', 'prev_pitching_IP']]

    def unroll_schedule(self, schedule, allow_missing=False):
        fields_general = ['game_id', 'game_datetime', 'umpire', 'matching_pitchers']
        fields_team = ['name', 'probable_pitcher', 'k', 'bb', 'ip',
                       'hitting_strikeOuts', 'hitting_baseOnBalls', 'hitting_atBats',
                       'pitching_strikeOuts', 'pitching_baseOnBalls', 'pitching_atBats',
                       'pitching_inningsPitched'
                       ]
        outcomes = []
        for team, not_team in (('home', 'away'), ('away', 'home')):
            fields_grab = fields_general + [f'{team}_{f}' for f in fields_team] + [f'{not_team}_name']
            fields_rn = {f'{team}_{f}': f for f in fields_team[2:]}
            fields_rn[f'{team}_name'] = 'pitching_team'
            fields_rn[f'{team}_probable_pitcher'] = 'pitcher'
            fields_rn[f'{not_team}_name'] = 'hitting_team'
            _outcomes = schedule[fields_grab].rename(columns=fields_rn)
            _outcomes['home_away'] = team
            outcomes.append(_outcomes)
        outcomes = pd.concat(outcomes)
        if allow_missing:
            outcomes = outcomes.drop(columns='matching_pitchers')
        else:
            outcomes = outcomes[outcomes.matching_pitchers].drop(columns='matching_pitchers')
        team_u = outcomes.pitching_team.unique()
        good = np.array([team in self.team_dict_map for team in team_u])
        bad_teams = team_u[~good]
        outcomes_u = outcomes[~(np.isin(outcomes.pitching_team, bad_teams) | np.isin(outcomes.hitting_team, bad_teams))]
        # self.schedule_unrolled = outcomes_u
        return outcomes_u

    @staticmethod
    def extract_schedule_stats(schedule_unrolled):
        k_ab = np.stack(schedule_unrolled.hitting_strikeOuts / schedule_unrolled.hitting_atBats)
        bb_ab = np.stack(schedule_unrolled.hitting_baseOnBalls / schedule_unrolled.hitting_atBats)
        # fill nans
        k_ab[np.isnan(k_ab) | np.isinf(k_ab)] = np.nanmean(k_ab[~np.isinf(k_ab)])
        bb_ab[np.isnan(bb_ab) | np.isinf(bb_ab)] = np.nanmean(bb_ab[~np.isinf(bb_ab)])

        schedule_unrolled['pred_k_hitting'] = k_ab @ pred_num_at_bat
        schedule_unrolled['pred_bb_hitting'] = bb_ab @ pred_num_at_bat

        schedule_unrolled['pitching_k_inning'] = (schedule_unrolled['pitching_strikeOuts']
                                                       / schedule_unrolled['pitching_inningsPitched'])
        schedule_unrolled['pitching_bb_inning'] = (schedule_unrolled['pitching_baseOnBalls']
                                                        / schedule_unrolled['pitching_inningsPitched'])
        return schedule_unrolled

    def shrink_schedule(self, no_drop=False):
        cat_types = ['home_name', 'away_name', 'home_probable_pitcher', 'away_probable_pitcher', 'umpire',
                     'matching_pitchers']
        int_types = ['home_k', 'home_bb', 'away_k', 'away_bb',
                     'home_pitching_strikeOuts', 'home_pitching_baseOnBalls', 'home_pitching_atBats',
                     'away_pitching_strikeOuts', 'away_pitching_baseOnBalls', 'away_pitching_atBats'
                     ]
        float_types = ['home_ip', 'away_ip',
                       'home_pitching_inningsPitched', 'away_pitching_inningsPitched']
        pass_types = ['game_id', 'game_datetime',
                      'home_hitting_strikeOuts', 'home_hitting_baseOnBalls', 'home_hitting_atBats',
                      'away_hitting_strikeOuts', 'away_hitting_baseOnBalls', 'away_hitting_atBats']

        for col in cat_types:
            self.schedule[col] = self.schedule[col].astype('category')

        for col in int_types:
            self.schedule[col] = self.schedule[col].astype('int16')

        for col in float_types:
            self.schedule[col] = self.schedule[col].astype('float16')
        if no_drop:
            bad_rows = (self.schedule.home_k == -555) # none...
        else:
            bad_rows = (self.schedule.home_k == -999)
        self.schedule = self.schedule.loc[~bad_rows, pass_types + cat_types + int_types + float_types]

    def shrink_merged_schedule(self, merged_schedule, additional_cols = None):
        if additional_cols is None:
            additional_cols = []
        merged_schedule['date_time'] = merged_schedule.game_datetime.apply(parser.parse)
        merged_schedule = merged_schedule.sort_values('date_time')
        merged_schedule['day'] = (
            (merged_schedule['date_time'] - merged_schedule['date_time'][0]).
            apply(lambda x: x.days).astype(np.int16))

        col_keep = ['game_id', 'pitching_team', 'hitting_team', 'date_time', 'day',
                    'pitcher', 'home_away', 'umpire',
                    'k', 'bb', 'ip',  # outcome
                    'pitching_inningsPitched', 'pitching_k_inning', 'pitching_bb_inning', # this season
                    'pred_k_hitting', 'pred_bb_hitting',  # this season
                    'prev_pitching_IP', 'prev_pitching_k_game', 'prev_pitching_bb_game','prev_pitching_ip_game', # last seasob
                    'prev_hitting_Kp','prev_hitting_BBp'] + additional_cols # last season
        merged_schedule_shrunk = merged_schedule.loc[:, col_keep].copy() # drop copy later..

        for col in self.col_to_cat:
            merged_schedule_shrunk[col] = merged_schedule_shrunk[col].astype('category')

        rename_dict = {'pitching_inningsPitched': 'P_IP', 'pitching_k_inning': 'P_k_inn',
                       'pitching_bb_inning': 'P_bb_inn',
                       'pred_k_hitting': 'H_k', 'pred_bb_hitting': 'H_bb',
                       'prev_pitching_IP': 'pP_IP', 'prev_pitching_k_game': 'pP_kg', 'prev_pitching_bb_game': 'pP_bbg',
                       'prev_pitching_ip_game': 'pP_IPg',
                       'prev_hitting_Kp': 'pH_kp', 'prev_hitting_BBp': 'pH_bbp'
                       }
        # self.merged_schedule_shrunk = self.merged_schedule_shrunk.rename(rename_dict)
        return merged_schedule_shrunk.rename(rename_dict)

    def save_name_schedule(self,season: int) -> str:
        return f'{self.loc_schedule}/{season}_schedule.parquet'
    def save_name_merged_schedule(self,season: int) -> str:
        return f'{self.loc_merged_schedule}/{season}_merged_schedule.parquet'
    @staticmethod
    @cache
    def boxscore_game_pk(game_pk):
        return statsapi.boxscore_data(game_pk)

    @staticmethod
    @cache
    def schedule_date(this_date=None):
        return statsapi.schedule(this_date)

    @classmethod
    def get_info_boxscore(cls, game_pk, ump_only=False, add_season_stats=True):
        out = cls.boxscore_game_pk(game_pk)
        box_info = out['gameBoxInfo']
        m = {}
        for item in box_info:
            m[item['label']] = item.get('value', '')
        umpires = m.get('Umpires', 'None')
        if umpires != '':
            if (umpires[-12:] == 'first pitch)') or (umpires[-13:] == 'first pitch).'):
                hp_ump = 'None'
            else:
                hp_ump = umpires.split('. 1B:')[0]
                assert hp_ump[:3] == 'HP:', 'ump string not expected format'
                hp_ump = hp_ump[4:]
        else:
            hp_ump = 'None'
        if ump_only:
            return hp_ump

        key_info = {'game_id': game_pk, 'umpire': hp_ump}
        for team in ['home', 'away']:
            try:
                this_d = out[f'{team}Pitchers'][1]
            except IndexError:
                this_d = {'namefield': 'None', 'k': -999, 'bb': -999, 'ip': np.nan}
            key_info[f'{team}_pitcher_name'] = this_d['namefield'].split(' ')[0].strip(',')
            key_info[f'{team}_k'] = int(this_d['k'])
            key_info[f'{team}_bb'] = int(this_d['bb'])
            key_info[f'{team}_ip'] = float(this_d['ip'])
            if add_season_stats:
                pitching_stats = cls.extract_pitching_stats(out[team])
                hitting_stats = cls.extract_hitting_stats(out[team])
                for k, v in pitching_stats.items():
                    key_info[f'{team}_pitching_{k}'] = v
                for k, v in hitting_stats.items():
                    key_info[f'{team}_hitting_{k}'] = v
        return key_info

    @staticmethod
    def extract_hitting_stats(team_table, stats_grab=('strikeOuts', 'baseOnBalls', 'atBats')):
        hitting_dict = {stat: np.zeros(9, dtype=np.uint16) for stat in stats_grab}
        # batting_stats_array = np.zeros((9, len(stats_grab)), dtype=np.uint16)
        if 'battingOrder' in team_table:
            these_players = team_table['battingOrder']
            for pi, player in enumerate(these_players):
                batting_stats = team_table['players'][f'ID{player}']['seasonStats']['batting']
                for i, stat in enumerate(stats_grab):
                    hitting_dict[stat][pi] = batting_stats[stat]
                    # batting_stats_array[pi, i] = batting_stats[stat]
        return hitting_dict

    @staticmethod
    def extract_pitching_stats(team_table, stats_grab=('strikeOuts', 'baseOnBalls', 'atBats', 'inningsPitched')):
        pitching_dict = {stat: -999 for stat in stats_grab}
        if len(team_table['pitchers']) >= 1:
            starting_pitcher = team_table['pitchers'][0]
            pitching_dict['name'] = team_table['players'][f'ID{starting_pitcher}']['person']['fullName']
            pitching_stats = team_table['players'][f'ID{starting_pitcher}']['seasonStats']['pitching']
            for stat in stats_grab:
                pitching_dict[stat] = pitching_stats[stat]
                if type(pitching_dict[stat]) is str:
                    pitching_dict[stat] = float(pitching_dict[stat])
        return pitching_dict

    @classmethod
    def get_schedule_for_date(cls, date=None, pull_boxscore=False, want_umpire=False):
        data = cls.schedule_date(date)
        schedule = pd.DataFrame(data, columns=['game_id', 'game_datetime', 'away_name', 'home_name',
                                               'home_probable_pitcher', 'away_probable_pitcher', 'current_inning','status'])

        if pull_boxscore: # does nothing?
            game_pks = schedule['game_id'].values
            [cls.get_info_boxscore(game_pk, True) for game_pk in game_pks]
        if want_umpire:
            game_pks = schedule['game_id'].values
            umps = [cls.get_info_boxscore(game_pk, True) for game_pk in game_pks]
            schedule['umpire'] = umps
        return schedule

    @classmethod
    def get_outcomes_from_game_dates(cls, dates_games):
        schedules = []
        for this_date in dates_games:
            print(this_date.strftime('%m-%d'), end=', ')
            _schedule = cls.get_schedule_for_date(this_date.strftime('%Y-%m-%d'), want_umpire=False)
            if len(_schedule) == 0:
                continue
            box_scores = pd.DataFrame(
                [cls.get_info_boxscore(game_id) for game_id in _schedule.game_id])
            _schedule_merged = _schedule.merge(box_scores, on='game_id')
            schedules.append(_schedule_merged)

        schedule = pd.concat(schedules)
        hpp = schedule.home_probable_pitcher
        app = schedule.away_probable_pitcher
        schedule['home_prob_pitch_ln'] = [_hpp.split(' ')[-1] for _hpp in hpp]
        schedule['away_prob_pitch_ln'] = [_app.split(' ')[-1] for _app in app]
        matching_pitchers = ((schedule['home_prob_pitch_ln'] == schedule['home_pitcher_name']) &
                             (schedule['away_prob_pitch_ln'] == schedule['away_pitcher_name']))
        schedule['matching_pitchers'] = matching_pitchers
        schedule = schedule.sort_values('game_datetime')
        return schedule

    @staticmethod
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)


class DataLoaderGame(ScheduleETL):
    def __init__(self):
        super().__init__()
        self.data = None

    def import_merged_data(self, shrink=True):
        fls = glob(self.loc_merged_schedule +'/*.parquet')
        data = []
        for fl in fls:
            _d = pd.read_parquet(fl)
            data.append(_d)
        self.data = pd.concat(data).reset_index(drop=True).sort_values('date_time')
        self.data['season'] = self.data.date_time.apply(lambda x: x.year)

        if shrink:
            self.data = self.shrink_data(self.data)
        return self.data

    def import_upcoming(self):
        self.upcoming_games = pd.read_parquet(self.loc_upcoming)
        return self.upcoming_games

    def shrink_data(self,data):
        for col in self.col_to_cat:
            data[col] = data[col].astype('category')
        data['game_id'] = data['game_id'].astype(int)
        return data