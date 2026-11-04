"""
Training modules of the Phantom VFL framework (James Mansueto's own code).

The Flower app in flower_app/ calls these modules:
  class_balance  class-balance record of a run (shares of the positive labels, loss values)
  flower_data    data folders, client features, scalers and labels
  losses         loss functions of the head model (bce, weighted_bce, focal, class_balanced)
  metrics        test and validation metrics of an evaluated round
  results        results folder and file names of a run
  seeding        seeds of the start weights of the client and head models

The launcher run_flower starts 'flwr run' for one run and copies its screen
output to <data-dir>/logs/ (framework.run_log). The runner run_training reads
run_config.json and calls the launcher for each seed and k level. The tuning
driver tune reads the tuning section of run_config.json, calls the launcher for
each trial, seed and k level, and ranks the trials on the validation metrics.
The app calls none of them.

The app finds this package through PYTHONPATH (the repository root).
"""
