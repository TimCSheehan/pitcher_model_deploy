import numpy as np, pandas as pd, seaborn as sns
from base_models import ModelFitting, ProbStrike # src.mlb_models.
from download_data import DataLoader as DL
import warnings
# noinspection PyUnresolvedReferences
import scipy.stats, scipy.optimize
import platform
dl = DL()


def get_data_root():
    if platform.system() == 'Darwin':
        return '/Users/tim/data/baseball/'
    elif platform.system() == 'Linux':
        return '/home/oddsbot16/data/test_baseball'
def sigmoid(x): return 1 / (1 + np.exp(-x))  # aka sigmoid
def softplus(x): return np.log(1 + np.exp(x))
link_sd = softplus

def link_cov(x): return sigmoid(x)*2-1


def gauss2_ll(params,loc,return_params=False):
    assert len(params)==5, 'incorrect params'
    mu = params[:2]
    _var = link_sd(params[2:4])
    _cov = link_cov(params[4])
    cov = np.array([[_var[0],_cov],[_cov,_var[1]]])
    if np.linalg.det(cov)<0:
        return - 1000
    if return_params:
        return mu,cov
    p_pitch = scipy.stats.multivariate_normal.pdf(loc, mu, cov)
    nll = -np.sum(np.log(p_pitch))
    return nll

def get_p_strike(data, iterate_over='season', n_sd=4, want_p_strike=True, sz_bt=False,verbose=True,
                 min_pitch_fit=20):
    p0 = dl.get_pitch_params_default(n_sd)
    var0 = sorted(data[iterate_over].unique())

    coef_all =  np.zeros((len(var0), 2, len(p0)))
    did_pass_all = np.zeros((len(var0), 2))
    query_str0 = f'{iterate_over}==@v0'
    query_str1 = query_str0 + ' and stand==@batter'
    for v0_i, v0 in enumerate(var0):
        for hi, batter in enumerate(('L', 'R')):
            _data = data.query(query_str0)
            _pitch_use = dl.get_pitch_use(_data, batter=batter)
            if min_pitch_fit and (np.sum(_pitch_use)<min_pitch_fit):
                # skip, not enough pitches to fit
                did_pass_all[v0_i, hi] = -1
                print(f'[skip]{v0_i + 1}/{len(var0)}', end=' ')
                continue
            _dat,_strike, _var_names, p0, _ = dl.get_pitch_data(_data, _pitch_use,n_sd=n_sd,
                                                             sz_bt=sz_bt)
            MF = ModelFitting(ProbStrike, data=(_dat,_strike), model_type='strike_zone',
                              param_names=_var_names, n_sd=n_sd, sz_bt=sz_bt,verbose=verbose)
            MF.fit_grad(p0)
            did_pass = check_coef(MF.labeled_coef,verbose)
            did_pass_all[v0_i,hi] = did_pass
            coef_all[v0_i, hi] = MF.fit_model.params
            if did_pass and want_p_strike:
                _pitch_test = (data[iterate_over] == v0) & (data.stand == batter)
                _d_use_all, _, _, _ = dl.get_pitch_data(_data, _pitch_test,n_sd=n_sd,
                                                             sz_bt=sz_bt)
                p_strike = MF.fit_model.p_strike(_d_use_all)
                data.loc[_pitch_test, 'pStrike'] = p_strike

        print(f'{v0_i + 1}/{len(var0)}', end=' ')
    return data, coef_all, list(MF.labeled_coef.keys()), did_pass_all, var0


acceptable_ranges = {'Left_Wall': (-1.2,0.7),
                     'Right_Wall': (0.7,1.2),
                     'Bottom_Wall': (1.0,2.0),
                     'Top_Wall': (3.0,3.8),
                     'L_Left_Wall': (-1.2,0.7),
                     'L_Right_Wall': (0.7,1.2),
                     'R_Left_Wall': (-1.2,0.7),
                     'R_Right_Wall': (0.7,1.2),
                     'Bottom Wall_alt': (-.5,.5),
                     'Top Wall_alt': (-.5, .5),
                     'SD': (-3,-1), }


def check_coef(labeled_coef,verbose):
    """
    Check if fit coeffiecients are within acceptable ranges. Helpful when fitting many models and want to
    maintain some quality control.
    """
    did_pass = True
    for nam,value in labeled_coef.items():
        nam_st = nam.split(' ')[0]
        nam_alt = nam+'_alt'
        if nam in acceptable_ranges:
            this_rng = acceptable_ranges[nam]
        elif nam_st in acceptable_ranges:
            this_rng = acceptable_ranges[nam_st]
        else:
            warnings.warn(f'Feature {nam} not marked')
            continue
        if nam_alt in acceptable_ranges:
            this_rng_alt = acceptable_ranges[nam_alt]
        else:
            this_rng_alt = None

        if (value < this_rng[0]) or (value > this_rng[1]):
            if this_rng_alt and ((value > this_rng_alt[0]) or (value < this_rng_alt[1])):
                continue
            if verbose:
                warn_str = f'Feature {nam} outside of acceptable range with value {value:.3f}'
                warnings.warn(warn_str)
            did_pass = False
    return did_pass


def get_split_figure(data, z_min=1.8, z_max=3.1, d_max=.6, n_bin0=30, n_bin1=12):
    bns = np.linspace(-1.6, 1.6, n_bin0 + 1)
    bn_labs = np.round((bns[:-1] + (bns[1] - bns[0]) / 2) * 100).astype(int)
    data['x_bin'] = pd.cut(data.plate_x.astype(float), bns, labels=bn_labs)

    bns = np.linspace(-1.6, 1.6, n_bin1 + 1)
    bn_labs = np.round((bns[:-1] + (bns[1] - bns[0]) / 2) * 100).astype(int)
    data['x_bin_prev'] = pd.cut(data.prev_plate_x.astype(float), bns, labels=bn_labs)

    _data_sub_prev = data.query('called_pitch and @z_min<plate_z<@z_max and d_euc < @d_max')
    this_agg_sub = _data_sub_prev.groupby(['x_bin', 'stand', 'x_bin_prev']).strike.mean().reset_index()
    sns.lineplot(data=this_agg_sub[this_agg_sub.stand == 'R'], x='x_bin', y='strike', hue='x_bin_prev', linestyle='--')