# Pitcher Model

This repository contains a script that is deployed as a Google Cloud Function and is currently scheduled to run daily (at 7AM). The entrypoint [`main.py`](https://github.com/TimCSheehan/pitcher_model_deploy/blob/main/main.py) triggers a sequence to get the current
days schedule and run a predictive model to predict the number of strike outs and walks thrown in each game by the starting pitcher. The model
predictions are then written to a database and displayed on my [personal website](timothysheehan.com/mlb_vis).

# Files
* [`exports.py`](https://github.com/TimCSheehan/pitcher_model_deploy/blob/main/exports.py) handles writing to the database and is also equipped to send a summary email upon model completion (currently deactivated).
* [`download_game_level_data.py`](https://github.com/TimCSheehan/pitcher_model_deploy/blob/main/download_game_level_data.py) handles the ETL pipeline
  for extracting game level variables including the season and historical stats of the pitcher, the pitchers team, the batting team, and the specific batters in the lineup.
  Relevent data is stored in a highly compressed format in the `/data` directory.
* [`main.py`](https://github.com/TimCSheehan/pitcher_model_deploy/blob/main/main.py) calls ETL pipeline and trains model on most up to data data to make predictions for todays game.
  Current best performing model is an implementation of Gradient Boosting Trees ([`xgboost`](https://xgboost.readthedocs.io/en/stable/)). To get an accurate estimate of model error,
  the model is first run holding out the previous two weeks of data and MSE is saved to a table before training on all past data to generate predictions. 
