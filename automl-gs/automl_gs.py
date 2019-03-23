import pandas as pd
from autopep8 import fix_code
from jinja2 import Environment, PackageLoader
from tqdm import tqdm
from datetime import datetime
import yaml
import os
import shutil
import uuid
from utils_automl import *


def automl_grid_search(csv_path, target_field,
                       target_metric=None,
                       framework='tensorflow',
                       model_name='automl',
                       context='standalone',
                       num_trials=100,
                       split=0.7,
                       num_epochs=20,
                       col_types={},
                       gpu=False,
                       tpu_address=None):
    """Parent function which performs the hyperparameter search.

    See the package README for parameter descriptions:
    https://github.com/minimaxir/automl-gs
    """

# Prepare environment and source data
env = Environment(
    loader=PackageLoader('automl_gs', 'templates'),
    trim_blocks=True,
    lstrip_blocks=True
)

df = pd.read_csv(csv_path, nrows=100)
object_cols = [col for col, col_type in df.dtypes.iteritems() if col_type == 'object']
df[object_cols] = df[object_cols].apply(pd.to_datetime, errors='ignore')

problem_type, target_metric, direction = get_problem_config(
    df[target_field], framework, target_metric)
input_types = get_input_types(df, col_types, target_field)
hp_grid = build_hp_grid(framework, input_types.values(), num_trials)
fields = normalize_col_names(input_types)

metrics_csv = open("automl_results.csv", 'w')
best_result = None
timeformat_utc = "{:%Y%m%d_%H%M%S}".format(datetime.utcnow())
best_folder = "{}_{}_{}".format(model_name, framework, timeformat_utc)
train_folder = "{}_train".format(model_name)
cmd = build_subprocess_cmd(csv_path, train_folder)

pbar = tqdm(hp_grid, smoothing=0, unit='trial')
for params in pbar:

    # Create destination folders for the model scripts + metadata
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
        os.mkdir(os.path.join(train_folder, 'metadata'))
        os.mkdir(os.path.join(train_folder, 'encoders'))

    # Generate model files according to the given hyperparameters.
    render_model(params, model_name,
                 framework, env, problem_type,
                 target_metric, target_field,
                 train_folder, fields, split, num_epochs, gpu, tpu_address)

    # Execute model training using the generated files.
    train_generated_model(cmd, num_epochs, train_folder)

    # Load the training results from the generated CSV,
    # and append to the metrics CSV.
    results = pd.read_csv(os.path.join(train_folder, 
                                       "metadata", "results.csv"))
    results = results.assign(**params)
    results.insert(0, 'trial_id', uuid.uuid4())

    results.to_csv("automl_results.csv", mode="a", index=False,
                   header=(best_result is None))

    train_results = results.tail(1).to_dict('records')[0]

    # If the target metric improves, save the new hps/files,
    # update the hyperparameters in console,
    # and delete the previous best files.

    if direction == 'max':
        top_result = results[target_metric].max()
    else:
        top_result = results[target_metric].min()

    if best_result is None:   # if first iteration
        best_result = top_result
        shutil.copytree(train_folder, best_folder)
        print_progress_tqdm(params, train_results, pbar, False)
    else:
        is_imp = best_result > top_result
        is_imp = not is_imp if direction == 'min' else is_imp
        if is_imp:
            shutil.rmtree(best_folder)
            shutil.copytree(train_folder, best_folder)
            print_progress_tqdm(params, train_results, pbar)

    # Clean up the generated file folder for the next trial.
    shutil.rmtree(train_folder)

metrics_csv.close()
pbar.close()
