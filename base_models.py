import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize

diameter_baseball = 2.9/12 # in feet
width_home_plate = 17/12
def sem(x):
    return np.std(x) / np.sqrt(len(x))
def sigmoid(x): return 1 / (1 + np.exp(-x))  # aka sigmoid
def logit(p, eps=1e-4):
    if type(p) is np.ndarray:
        p[p < eps] = eps
        p[p > (1 - eps)] = (1 - eps)
    return np.log(p / (1 - p))
def softmax(x, axis=1):
    ex = np.exp(x)
    return ex / (np.sum(ex, axis))[:, np.newaxis]
def softplus(x): return np.log(1 + np.exp(x))
def inv_softplus(x): return np.log(np.exp(x) - 1)


def modify_p_strike_logit(p_all,logit_model):
    tp = np.exp(logit_model[1] * logit(p_all) + logit_model[0])
    return tp/(1+tp)

def relu(direction, p, xx):
    y = xx.copy()
    amp, offset = p
    closest_ind = np.argmin(np.abs(y-offset))
    if direction > 0:
        y[closest_ind:] -= y[closest_ind]
        y[:closest_ind] = 0
    else:
        y[:closest_ind] -= y[closest_ind]
        y[closest_ind:] = 0
    y *= amp
    return y

def relu_sym(p0, xx, direction = -1):
    p1 = (p0[0], -p0[1])
    y0 = relu(direction, p0, xx)
    y1 = relu(-direction, p1, xx)
    return y0, y1

def f1_score(Y,Y_hat):
    Y_hat = Y_hat>0.5
    TP = np.sum(Y&Y_hat)
    FP = np.sum(~Y&Y_hat)
    FN = np.sum(Y&~Y_hat)
    return 2*TP/(2*TP+FP+FN)

def get_coef_prob(_W,intercept=None):
    """ Assumes first W is intercept, may want to enforce more formally"""
    W = _W.copy()
    if intercept is None:
        intercept = _W[0]
        W = W[1:]
    these_probs = np.exp(W + intercept)
    these_probs = these_probs / (1 + these_probs)
    return these_probs


dict_params = {'walls_single': ['Left_Wall', 'Right_Wall', 'Bottom_Wall', 'Top_Wall'],
               'walls_batter': ['L_Left_Wall', 'L_Right_Wall', 'Bottom_Wall', 'Top_Wall', 'R_Left_Wall', 'R_Right_Wall'],
               'sd_1': ['SD_ALL'],
               'sd_2': ['SD X', 'SD Y'],
               'sd_4': ['SD Left', 'SD Right', 'SD Bottom', 'SD Top'],
               'log_weight': ['p_strikeZone', 'p_ball']}

# TODO: don't like how sz takes list of rows and logistic model takes columns of data. They should look similar

class ProbStrike:
    """
    Defines a model for predicting the probability of a strike given a pitch location.
    Includes varients for logistic regression and joint models. Model keeps track of parameters
    and column names of data. Coefficient estimation achieved with ModelFitting class.
    """

    def __init__(self, model_type='strike_zone', params=None, n_sd=4, param_names=None,
                 mod_wall_relu=None, sz_bt=False, batter_lr=False):

        self.model_type = model_type  # {strike zone, history, more...}
        self.params = params
        self.link_sd = softplus
        self.params_sz = None
        self.W = None
        self.n_sd = n_sd
        self.include_sz_bot_top = sz_bt
        self.batter_lr = batter_lr
        wall_use = ['walls_single', 'walls_batter'][1*batter_lr]
        self.n_wall = len(dict_params[wall_use])

        self.relu_pivot = None
        self.mod_wall_relu = False
        # TODO: check that this doesn't break RELU when wanted. Values should be updated in self.handle_relu_params
        add_relu_params = self.handle_relu_params(mod_wall_relu)

        if self.model_type == 'strike_zone':
            self.params_sz = self.params
            self.p_strike = self.strike_zone_sd
            self.parameter_names = dict_params[wall_use] + dict_params[f'sd_{self.n_sd}']
            if self.mod_wall_relu:
                self.parameter_names += add_relu_params
            self.loss_fun = self.lik_sz_joint
        elif self.model_type == 'logistic':
            self.n_sd = 0
            self.W = params
            self.p_strike = self.p_strike_logistic
            self.loss_fun = self.logistic_loss_fun
            self.parameter_names = param_names

        elif self.model_type == 'joint':
            self.do_all_logistic = True
            assert param_names[-1] == 'strike_zone', 'Must include pStrike column to overwrite in X'
            self.params_sz = self.params[:self.n_wall + n_sd + len(add_relu_params)]
            self.W = self.params[self.n_wall + n_sd + len(add_relu_params):]
            self.p_strike = self.p_strike_joint_model
            self.loss_fun = self.joint_sz_log_loss
            self.parameter_names = (dict_params[wall_use] + dict_params[f'sd_{self.n_sd}'] +
                                    add_relu_params + param_names)
        else:
            raise ValueError('Not a valid model')

        if param_names and (self.W is not None): assert len(param_names) == len(self.W), 'invalid param names'
        self.bits_gained_model = None

    def strike_zone_sd(self, X, wall_prob=False):
        """
        Function to provide pStrike(x,y).
        Handles
        - parameter sharing for SD
        - modifying wall based on history  (left/right wall)
        - modifying wall based on batter height  (top/bottom wall)
        - switching wall parameters based on batter handedness (left/right wall)
        """
        total_expected_n_params = (self.n_wall + self.n_sd + 1*self.mod_wall_relu)
        assert len(self.params_sz) == total_expected_n_params, 'invalid setup!'
        walls = self.params_sz[:self.n_wall]
        sd = self.link_sd(self.params_sz[self.n_wall:self.n_wall + self.n_sd])
        if self.mod_wall_relu:
            x, y, x_prev = X[0], X[1], X[-1]
            walls = self.mod_walls_relu_fun(walls, x_prev)
        else:
            x, y = X[0], X[1]

        if self.include_sz_bot_top:
            box_bot, box_top = X[2], X[3]
            walls = self.mod_box_top_bottom(walls, box_bot, box_top)

        # address hitter walls...
        if self.batter_lr:
            batter_right = X[-1-1*self.mod_wall_relu]
            walls = self.mod_walls_batter(walls,batter_right)

        p_all = self.prob_box(walls, sd, x, y)
        if wall_prob:
            p_walls = self.prob_box(walls, sd, x, y, wall_prob=1)
            return p_all, p_walls
        return p_all

    def mod_walls_batter(self, walls, batter_right):
        assert len(walls) == 6, 'Something wrong here!'
        if type(walls) is list or (walls.ndim == 1):
            walls = np.tile(np.expand_dims(walls, 1), [1, len(batter_right)])
        walls[:2, batter_right] = walls[-2:, batter_right]
        return walls[:4]


    def mod_walls_relu_fun(self, walls, x_prev):
        p = self.params_sz[self.n_wall + self.n_sd:]
        if len(p) == 1:
            p = [p, self.relu_pivot]
        mod_left, mod_right = relu_sym(p, x_prev)
        walls_new = np.tile(np.expand_dims(walls, 1), [1, len(mod_left)])
        walls_new[0] += mod_left
        walls_new[1] += mod_right
        if self.n_wall == 6:
            walls_new[4] += mod_left
            walls_new[5] += mod_right
        return walls_new

    @staticmethod
    def mod_box_top_bottom(walls, box_bot, box_top):
        if type(walls) is list or (walls.ndim == 1):
            walls = np.tile(np.expand_dims(walls, 1), [1, len(box_top)])
            walls[2] += box_bot
            walls[3] += box_top
        else:
            # already expanded re RELU
            walls[2] += box_bot
            walls[3] += box_top
        return walls

    def lik_sz_joint(self, X, strike, want_lik=False):
        """
        Provide likelihood for joint strike zone. Note that this function has a compressive shortcut
        To avoid 0 prob events. Will want to remove when combining with logistic function.
        """
        p_this = self.p_strike(X)
        loss = strike * p_this + (1 - strike) * (1 - p_this)
        loss_compress = 0.98 * loss.astype(float) + .01
        j = -np.sum(np.log(loss_compress))
        if want_lik: return -j
        return j

    def p_strike_logistic(self, x):
        p_strike_log = sigmoid(x @ self.W)
        return p_strike_log

    def logistic_loss_fun(self, x, y, lam=0, want_lik=False):
        # Has built in L2 regularization
        # need to make y [-1/+1], not 0,1
        if np.unique(y)[0] == 0:  # fix, should be asserted or applied ahead of time to save time in function
            y = (y * 2) - 1
        a = -y * (x @ self.W).ravel()
        # loss_fit = np.sum(np.log(np.exp(a) + 1)) # neg log-likelihood. Can drop 1/ since in log (just minus sign).
        if want_lik: return -np.sum(np.log(np.exp(a) + 1))
        loss_fit = np.mean(np.log(np.exp(a) + 1)) # mean so scaling of c is relatively constant x dataset size
        if lam == 0:
            return loss_fit
        else:
            loss_w = self.W.T @ self.W
            loss_total = loss_fit + lam*loss_w
        return loss_total

    def p_strike_joint_model(self, Xs, Xw):
        assert len(self.W) == Xw.shape[1], 'Parameter Count off!'
        if self.do_all_logistic:
            Xw[:, -1] = logit(self.strike_zone_sd(Xs))
            this_prob = self.p_strike_logistic(Xw)
        else:
            p_sz, p_ball = sigmoid(self.model_weighting)
            p_ball = p_ball * (1 - p_sz)  # ensure these cannot sum greater than 1
            p_strike_sz = self.strike_zone_sd(Xs)
            p_strike_logistic = self.p_strike_logistic(Xw)
            this_prob = p_sz * p_strike_sz + p_ball * 0 + (1 - p_sz - p_ball) * p_strike_logistic
        return this_prob

    def joint_sz_log_loss(self, Xs, Xw, Y, lam=0, want_lik=False):
        """
        params (jointSZ,(pSZ,pBall),logistic function)
        xx,yy  (plate position)
        hand   (handedness of batter, can set to None)
        X      (samples, feat) - history regressors
        Y      (samples, ) - called strike
        """
        # assert type(Y) is int, 'Y must be int'
        # Y = Y.astype(int)
        this_prob = self.p_strike_joint_model(Xs, Xw)

        this_lik = Y * this_prob + (1 - Y) * (1 - this_prob)
        assert np.all(this_lik > 0)  # better to kill early.
        this_ll = -np.sum(np.log(this_lik))
        if want_lik: return -this_ll
        if lam:
            w_penalty = (lam / 2) * np.dot(self.W, self.W)
            this_ll = this_ll + w_penalty
        return this_ll

    def coef_prob(self):
        if self.W is None:
            raise ValueError('Logistic Model not fit!')
        return get_coef_prob(self.W)


    @staticmethod
    def prob_box(walls, sd, x, y, for_vis=0, wall_prob=0):
        if len(sd) == 2:
            sd = [sd[0], sd[0], sd[1], sd[1]]
        elif len(sd) == 1:
            sd = sd * 4
        sd = np.expand_dims(sd, 1)
        if type(walls) is list or (walls.ndim == 1):
            walls = np.expand_dims(walls, 1)

        p_outside_wall_x = norm.cdf(walls[:2], x, sd[:2])  # not sure how dimensions will work here...
        p_outside_x = (1 - p_outside_wall_x[0]) * p_outside_wall_x[1]
        p_outside_wall_y = norm.cdf(walls[2:], y, sd[2:])
        p_outside_y = (1 - p_outside_wall_y[0]) * p_outside_wall_y[1]
        p_all = (p_outside_x * p_outside_y)
        if for_vis:
            strike_all = np.outer(p_outside_y, p_outside_x)
            return strike_all
        if wall_prob:
            return np.c_[1 - p_outside_wall_x[0], p_outside_wall_x[1], 1 - p_outside_wall_y[0], p_outside_wall_y[1]]
        return p_all

    @staticmethod
    def bits_per_trial(Y, LL_test):
        assert np.all(np.unique(Y) == [0, 1]), 'Y needs to be a Boolean'
        assert LL_test < 0, 'LL needs to be negative'
        p_null = np.sort(np.array([np.mean(Y), 1 - np.mean(Y)]))
        ns = np.sort(np.array([np.sum(Y == 0), np.sum(Y == 1)]))
        ll_null = ns @ np.log(p_null)  # probability of call given base rate

        bits_pt = (LL_test - ll_null) / (len(Y) * np.log(2))
        return bits_pt

    def bits_mdl(self, data):
        lik_mdl = self.loss_fun(*data, want_lik=True)
        return self.bits_per_trial(data[-1]==1, lik_mdl)

    def handle_relu_params(self, mod_wall_relu):
        add_relu_params = []
        if mod_wall_relu is None:
            self.mod_wall_relu = False
        elif type(mod_wall_relu) is dict:
            add_relu_params = mod_wall_relu['params']
            self.relu_pivot = mod_wall_relu.get('relu_pivot', False)
            self.mod_wall_relu = True
        elif mod_wall_relu is True:
            add_relu_params = ['relu_amp']
            self.relu_pivot = 0.0
            self.mod_wall_relu = True
        return add_relu_params


class ModelFitting:
    """
    Designed to fit a model defined by ProbStrike class. Can fit a single model or perform cross-validation.
    Returns many metrics including bits gained and F1-score.
    """
    def __init__(self, model, data, scale_coef=None, verbose=True, **kwargs):
        self.labeled_coef = None
        self.opts = {'maxiter': 200}
        self.minimization_approach = 'L-BFGS-B' #'BFGS'
        self.model = model
        self.kwargs = kwargs
        self.data = data
        self.do_scale_coef = scale_coef is not None
        self.scale_coef = scale_coef

        self.fit_model = None
        self.loss = None
        self.fit_object = None
        self.do_cv = False
        self.train_split = None
        self.this_G=None
        self.verbose = verbose
        # print(self.kwargs, len(self.data))
    def get_loss_model(self, params, *args):
        this_model = self.model(params=params, **self.kwargs)
        return this_model.loss_fun(*self.get_data(), *args)

    def get_data(self, get_test_split=False):
        if self.do_cv is False:
            return self.data
        elif get_test_split:
            these_ind = self.train_split == self.this_G
        else:
            these_ind = self.train_split != self.this_G
        data_out = []
        for d in self.data:
            try:
                _d = d[these_ind]
            except TypeError:
                _d = [dd[these_ind] for dd in d]
            data_out.append(_d)
        return data_out

    def fit_grad(self, p0=None, *args):
        # #-  fit model
        out = scipy.optimize.minimize(self.get_loss_model, p0, args=args, method=self.minimization_approach,
                                      options=self.opts)

        self.fit_model = self.model(params=out.x, **self.kwargs)
        self.loss = out.fun
        self.fit_object = out
        x,y = self.get_data()[:-1], self.get_data()[-1]
        y_hat = self.fit_model.p_strike(*x)
        f1 = f1_score(y, y_hat)
        self.labeled_coef = {self.fit_model.parameter_names[i]: out.x[i] for i in range(len(out.x))}
        bits_model = self.fit_model.bits_mdl(self.get_data())
        if self.verbose: print(f'Bits gained: {bits_model :.3f}. F1-score: {f1 :.3f}')
        return out

    def fit_grad_cv(self, p0, g=10, *args):
        n_row = len(self.data[-1])
        if type(g) is int:
            train_split = self.get_train_test_split(n_row, g)
            n_group = g
        else:
            assert len(self.data[-1]) == len(g), 'G not valid form!'
            train_split = g
            assert ~np.any(np.isnan(train_split)), 'Split messed up'
            n_group = len(np.unique(train_split))
        self.train_split = train_split
        self.do_cv = True # essential - impacts get_data

        cv_outputs = {'f1_score': [], 'bits_train': [], 'bits_test': [],
                      'labeled_coef': [], 'loss': [], 'fit_models': [], 'fitting_object': []}
        for G in range(n_group):
            if self.verbose: print(f'{G+1}/{n_group}', end=' ')
            self.this_G = G+1
            out = scipy.optimize.minimize(self.get_loss_model, p0, args=args, method=self.minimization_approach,
                                          options=self.opts)
            fit_model = self.model(params=out.x, **self.kwargs)
            cv_outputs['bits_train'].append(fit_model.bits_mdl(self.get_data()))
            cv_outputs['bits_test'].append(fit_model.bits_mdl(self.get_data(True)))
            x, y = self.get_data(True)
            y_hat = fit_model.p_strike(x)
            cv_outputs['f1_score'].append(f1_score(y, y_hat))
            cv_outputs['loss'].append(out.fun)
            cv_outputs['fit_models'].append(fit_model)
            cv_outputs['labeled_coef'].append({fit_model.parameter_names[i]: out.x[i] for i in range(len(out.x))})
            cv_outputs['fitting_object'].append(out)
        self.do_cv = False
        return cv_outputs

    @staticmethod
    def get_train_test_split(n, g):
        sz_block = np.floor(n / g)
        b_start = np.arange(0, n, sz_block, dtype=int)
        train_split = np.zeros(n, dtype=int)
        for i in range(g):
            if i == g - 1:
                train_split[b_start[i]:] = i + 1  # make sure we include all values
            else:
                train_split[b_start[i]:b_start[i + 1]] = i + 1
        assert ~np.any(np.isnan(train_split)), 'Split messed up'
        return train_split


class Visualization:
    """
    Helper class for visualizing fit models including strike zone and history features.
    """
    def __init__(self):
        pass

    @staticmethod
    def sz_plot(tight=False):
        if tight:
            xl, yl = (-2, 2), (1, 4)
        else:
            xl, yl = (-3, 3), (-2, 7)
        plt.xlim(xl)
        plt.ylim(yl)

    @staticmethod
    def gather_parameters_logistic(fit_model, prob_technique='1', p_strike=None, want_coef=False):

        if type(fit_model) is dict: # assuming labeled_coef
            param_names = list(fit_model.keys())
            coefs = list(fit_model.values())
            raise ValueError('Never updated indicies here, check param names and length of coefs...')
        else:
            param_names = fit_model.parameter_names
            if fit_model.params_sz is not None:
                param_names = param_names[len(fit_model.params_sz):]
            coefs = fit_model.W

        n_step_back = max([int(p[1]) if p[0] == 'B' else 0 for p in param_names])
        get_future = any([1 if p[0] == 'F' else 0 for p in param_names])

        outcome_coef = np.zeros((3, n_step_back + get_future)) * np.nan  # can be inferred from coef names...
        count_coef = np.zeros((4, 3)) * np.nan

        outcome_prob = np.zeros_like(outcome_coef)*np.nan
        count_prob = np.zeros_like(count_coef)*np.nan

        coef_ball_strike = np.zeros(2)*np.nan
        ind_map = {'ball': 0, 'called_strike': 1, 'swinging_strike': 2}
        ind_map_present = set()

        settings = {'ind_map_present':ind_map_present,
                    'n_step_back':n_step_back,
                    'get_future':get_future}

        intercept = coefs[0]
        for vi, v in enumerate(param_names):
            if v == 'balls':
                coef_ball_strike[0] = coefs[vi]
            elif v == 'strikes':
                coef_ball_strike[1] = coefs[vi]
            elif v == 'strike_zone':
                coef_strike_zone = coefs[vi]
            elif v[0] == 'B':
                ind, outcome = v.split('-')
                outcome_coef[ind_map[outcome], int(ind[1:]) - 1] = coefs[vi]
                ind_map_present.add(outcome)
            elif v[0] == 'F':
                ind = -1
                _, outcome = v.split('-')
                outcome_coef[ind_map[outcome], ind] = coefs[vi]
            elif v[:5] == 'count':
                b, s = v[6:].split('-')
                count_coef[int(b), int(s)] = coefs[vi]

        # convert coef to probabilities
        if prob_technique == '1':
            offset = intercept
        elif prob_technique == 'strike_zone':
            assert p_strike, 'need to include base rate'
            offset = intercept + p_strike*coef_strike_zone
        else:
            raise ValueError('Invalid probabilty calculating approach')

        if ~np.isnan(coef_ball_strike[0]):
            for b in range(4):
                for s in range(3):
                    this_prob = np.exp(offset + coef_ball_strike[0] * b + coef_ball_strike[1] * s)
                    count_coef[b,s] = coef_ball_strike[0] * b + coef_ball_strike[1] * s
                    count_prob[b, s] = this_prob / (1 + this_prob)
        else:
            count_prob = get_coef_prob(count_coef,intercept=offset)
        outcome_prob = get_coef_prob(outcome_coef,intercept=offset)

        if want_coef:
            return outcome_coef, count_coef, settings
        return outcome_prob, count_prob, settings

    @classmethod
    def vis_logistic_model(cls, outcome_prob, count_prob, settings, want_coef=False,
                           do_lines=True, title=None, share_y=True, y_lim=None, n_panel=2, ls='-',
                           marker='o'):

        if want_coef:
            ylab = r'$\beta$'
        else:
            ylab = 'P(Strike)'
        share_y = share_y and do_lines
        if share_y and (not y_lim):
            if outcome_prob.ndim==3:
                _outcome_prob = np.nanmean(outcome_prob,2)
                _count_prob = np.nanmean(count_prob,2)
            else:
                _outcome_prob = outcome_prob
                _count_prob = count_prob
            y_lim = (min(np.nanmin(_outcome_prob), np.nanmin(_count_prob)),
                     max(np.nanmax(_outcome_prob), np.nanmax(_count_prob)))

        plt.subplot(1, n_panel, 1)
        nb_x = np.arange(-1, -settings['n_step_back'] - 1, -1)
        # cols = ('b', 'r', 'g')
        cols = ('c', 'm', 'g')
        for i, nam in enumerate(settings['ind_map_present']):
            _outcome_prob = outcome_prob[i].copy()
            if settings['get_future']:
                if outcome_prob.ndim == 3:
                    cls.sem_plot(nb_x, outcome_prob[-1], color= cols[i], marker='x',eBar=True)
                else:
                    plt.plot(nb_x, outcome_prob[-1], color= cols[i], marker='x')
                _outcome_prob = outcome_prob[:-1]
            if outcome_prob.ndim == 3:
                cls.sem_plot(nb_x, _outcome_prob, color=cols[i], marker=marker, eBar=True, label=nam)
            else:
                plt.plot(nb_x, _outcome_prob, color=cols[i], label=nam, marker=marker,linestyle=ls)

        plt.ylabel(ylab)
        plt.xlabel('Relative Pitch')
        plt.legend(title='Previous Call')
        plt.xticks(nb_x)
        if share_y: plt.ylim(y_lim)
        plt.subplot(1, n_panel, 2)

        if do_lines:
            cols_strikes = [(0.70, 0.0, 0.0), (1.0, 0.36, 0.0), (1.0, 1.0, 0.03)]
            n_balls = np.arange(4)
            for i in range(3):
                if outcome_prob.ndim == 3:
                    cls.sem_plot(n_balls, count_prob[:, i], color=cols_strikes[i], marker=marker, eBar=True, label=i)
                else:
                    plt.plot(n_balls, count_prob[:, i], color=cols_strikes[i], marker=marker, label=i, linestyle=ls)
            plt.xlabel('# Balls')
            plt.legend(title='# Strikes')
            plt.ylabel(ylab)

        else:
            if count_prob.ndim == 3:
                count_prob = np.nanmean(count_prob,2)
            plt.imshow(count_prob.T, origin='lower', extent=(0, 4, 0, 3), cmap='hot')
            yl = plt.colorbar(label='P(Strike)')
            plt.xticks(np.arange(0.5, 3.6, 1), np.arange(4))
            plt.yticks(np.arange(0.5, 2.6, 1), np.arange(3))
            plt.xlabel('# Balls')
            plt.ylabel('# Strikes')
        if title:
            plt.title(title)
        if share_y:
            plt.ylim(y_lim)
        plt.tight_layout()

    @classmethod
    def vis_logistic_from_model(cls,fit_model, want_coef=False, **args_vis):
        outcome_prob, count_prob, settings = cls.gather_parameters_logistic(fit_model,want_coef=want_coef)
        cls.vis_logistic_model(outcome_prob, count_prob, settings, want_coef=want_coef, **args_vis)

    @classmethod
    def vis_logistic_stack(cls,stack_coef,  want_coef = False, return_stack=False, **kwargs):
        outcome_prob, count_prob = [], []
        for coef in stack_coef:
            _outcome_prob, _count_prob, settings = cls.gather_parameters_logistic(coef,want_coef=want_coef)
            outcome_prob.append(_outcome_prob)
            count_prob.append(_count_prob)
        outcome_prob = np.stack(outcome_prob,2)
        count_prob = np.stack(count_prob,2)
        if return_stack: return outcome_prob, count_prob, settings

        cls.vis_logistic_model(outcome_prob,count_prob, settings, want_coef=want_coef, **kwargs)


    @staticmethod
    def draw_box(rect=(-.93, .93, 1.6, 3.4), sty='k', lw=4, **args):
        plt.plot((rect[0], rect[1]), (rect[2], rect[2]), sty, linewidth=lw, **args)
        plt.plot((rect[0], rect[1]), (rect[3], rect[3]), sty, linewidth=lw, **args)
        plt.plot((rect[0], rect[0]), (rect[2], rect[3]), sty, linewidth=lw, **args)
        plt.plot((rect[1], rect[1]), (rect[2], rect[3]), sty, linewidth=lw, **args)

    @classmethod
    def vis_prob_box(cls, model, model_logit=False, **args):
        logit_model = None
        if hasattr(model,'fit_model'):
            model_strike = model.fit_model
            labeled_coef = model.labeled_coef
            if model_logit:
                logit_model = [labeled_coef['1'],labeled_coef['strike_zone']]
        else:
            model_strike = model
            if model_logit: raise ValueError('CANT USE LOGIT. Need to supply model fitted object, not fit_model')

        out = cls._vis_prob_box(model_strike.params_sz, model_strike.n_sd, model_strike.n_wall, logit_model=logit_model,
                                link_sd=model_strike.link_sd, **args)
        if out is not None:
            return out

    @staticmethod
    def _vis_prob_box(params, n_sd, n_wall=4, link_sd=None, scale_baseball=False, give_p_strike=False, logit_model=None, **args):
        rng_vis = (-2, 2, 0, 5)
        n_vis = 100
        two_batters = n_wall == 6
        xx = np.linspace(rng_vis[0], rng_vis[1], n_vis)
        yy = np.linspace(rng_vis[2], rng_vis[3], n_vis)
        grid = np.meshgrid(xx, yy)

        walls, sd = params[:n_wall], params[n_wall:n_wall + n_sd]
        # if give_p_strike:
        #     return walls,sd,xx,yy

        if scale_baseball:
            ad = diameter_baseball/2
            if two_batters:
                walls = [walls[0] - ad, walls[1] + ad, walls[2] - ad, walls[3] + ad, walls[4] - ad, walls[5] + ad]
            else:
                walls = [walls[0]-ad, walls[1]+ad, walls[2]-ad, walls[3]+ad]
        if link_sd:
            sd = link_sd(sd)

        if two_batters:
            walls_iter = [walls[:4], np.r_[walls[4:], walls[2:4]]]
            cols = ('b', 'r')
            labs = ('Left', 'Right')
        else:
            cols, labs = [args.get('color','')], [args.get('label','')]
            walls_iter = [walls]
        for i in range(len(walls_iter)):
            p_all = ProbStrike.prob_box(walls_iter[i], sd, xx, yy, for_vis=1)
            if logit_model:
                p_all = modify_p_strike_logit(p_all,logit_model)
            if give_p_strike:
                return p_all
            cs = plt.contour(*grid, p_all, levels=[0.5], colors='w')  # ,label=lab)
            p = cs.collections[0].get_paths()[0]
            v = p.vertices.T
            args['label'] = labs[i] #+ args.get('label', '')
            if cols[i]:
                args['color'] = cols[i]
            plt.plot(v[0], v[1], **args)
        plt.ioff()
        Visualization.sz_plot(True)

    @staticmethod
    def sem_plot(x, y, axs=0, within_E=0, do_line=1, outline=0, eBar=0, **kwargs):
        # x is assumed to be examples x len(y)
        keys_allow = ['color','label','marker']
        def filter_fun(kv): return kv[0] in keys_allow
        kwargs = dict(filter(filter_fun, kwargs.items()))
        n_ex, n_points = y.shape
        if n_points != len(x):
            y = y.T
            n_ex, n_points = y.shape
        assert n_points == len(x), 'x not correct shape'
        if np.any(np.isnan(y)):
            print('ignoring nan values!')
            n_ex = np.sum(~np.isnan(y), 0)
        m_y = np.nanmean(y, 0)

        if within_E:
            y_use = (y.T - np.mean(y, 1)).T
            s_y = np.nanstd(y_use, 0) / np.sqrt(n_ex)
        else:
            s_y = np.nanstd(y, 0) / np.sqrt(n_ex)
        if axs == 0:
            if eBar:
                plt.errorbar(x, m_y, s_y, **kwargs)
            elif outline:
                plt.plot(x, m_y + s_y, **kwargs)
                plt.plot(x, m_y - s_y, **kwargs)
            else:
                plt.fill_between(x, m_y - s_y, m_y + s_y, **kwargs)
                if do_line:
                    plt.plot(x, m_y, 'k')
        else:
            if outline:
                axs.plot(x, m_y + s_y, **kwargs)
                axs.plot(x, m_y - s_y, **kwargs)
            else:
                axs.fill_between(x, m_y - s_y, m_y + s_y, **kwargs)
                if do_line:
                    axs.plot(x, m_y, 'k')

    @staticmethod
    def draw_home_plate(y_loc=1.2, x_loc=0):
        pm = np.array([-1, 1])
        pp = np.ones(2)
        plt.plot(pm * width_home_plate / 2 + x_loc, pp * (y_loc), 'k', linewidth=3)

    @staticmethod
    def draw_ellipse(diameter_x, diameter_y=None, loc=(0, 0), col='k'):
        if diameter_y is None:
            diameter_y = diameter_x
        xx = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        plt.plot(diameter_x / 2 * xx.real + loc[0], diameter_y / 2 * xx.imag + loc[1], col)

    draw_circle = draw_ellipse
