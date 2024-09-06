from base_models import ProbStrike, InvalidInputError
import unittest
import numpy as np


# dl = DataLoader()

# test_data = 'functions/data/statcast/2017-month-06-06.parquet'
# test_data = dl.loc_statcast + '2017-month-06-06.parquet'
# data = DataLoader().import_data(fn=test_data,) # test data to use
# rule_book_wall = [-0.83, 0.83, 1.53, 3.37] # stable approximation of the strike zone

big_sz_2_sd = [-1, 1, 1, 5]
ex_pitch_loc, ex_call = [[0,3],[5,0],[-1,3],[0,1],[0,5]], [1,0,0.5,0.5,0.5]

ex_logistic_values_even = np.array([[0,0,0],[1,1,1],[0.5,0.5,0.5]])


class test_sz_models(unittest.TestCase):

    def test_p_strike_2_sd(self):
        """ 2 SD model should give reasonable values for pitch in middle and way outside """
        simple_sz = ProbStrike(params=big_sz_2_sd + [-3]*2,n_sd=2)
        for pitch_loc, call in zip(ex_pitch_loc, ex_call):
            self.assertAlmostEqual(simple_sz.p_strike(pitch_loc), call)

    def test_p_strike_4_sd(self):
        """ 4 SD model should give reasonable values for pitch in middle and way outside """
        simple_sz = ProbStrike(params=big_sz_2_sd + [-3]*4,n_sd=4)
        for pitch_loc, call in zip(ex_pitch_loc, ex_call):
            self.assertAlmostEqual(simple_sz.p_strike(pitch_loc), call)

    
    def test_parameter_number(self):
        """ Ensure that the correct number of parameters is asserted for sz model"""
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*6,n_sd=4)
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*8,n_sd=2)
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*8,n_sd=1)
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*4,n_sd=4)
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*10,n_sd=4)

    def test_parameter_types(self):
        """ Ensure that the correct number of parameters is asserted for sz model"""
        self.assertRaises(InvalidInputError,ProbStrike,params='hellos',n_sd=2)
        self.assertRaises(InvalidInputError,ProbStrike,params=['hello']*8)
        self.assertRaises(InvalidInputError,ProbStrike,params='hello',model_type='logistic')



class test_logistic_models(unittest.TestCase):
    def test_logistic_params_cancel_out(self):
        """ W and X values that should cancel out so pStrike=0.5 """

        mdl_logistic = ProbStrike(params=np.array([-1,0,1]),model_type='logistic')
        for ex_values in ex_logistic_values_even:
            self.assertAlmostEqual(mdl_logistic.p_strike(ex_values),0.5)

    def test_param_names_check(self):
        """ Ensure that the correct number of param names matches params"""
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*3,param_names=['test',]*2,model_type='logistic')
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*3,param_names=['test',]*4,model_type='logistic')
        self.assertRaises(InvalidInputError,ProbStrike,params=[0]*4,param_names=['test',]*3,model_type='logistic')

if __name__ == '__main__':
    unittest.main()



