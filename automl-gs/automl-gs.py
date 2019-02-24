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
                       model_name='automl',
                       framework='tensorflow',
                       context='standalone',
                       num_trials=1000,
                       col_types={},
                       **kwargs):
    """Parent function which performs the hyperparameter search.
    """

# Prepare environment and source data


env = Environment(
    loader=PackageLoader('automl-gs', 'templates'),
    trim_blocks=True,
    lstrip_blocks=True
)

df = pd.read_csv(csv_path)
object_cols = [col for col, col_type in df.dtypes.iteritems() if col_type == 'object']
df[object_cols] = df[object_cols].apply(pd.to_datetime, errors='ignore')

problem_type, target_metric, direction = get_problem_config(
    df[target_field], **kwargs)
input_types = get_input_types(df, col_types)
hp_grid = build_hp_grid(framework, input_types.values(), num_trials)

fields_norm = normalize_col_names(df)
df.columns = fields_norm

pbar = tqdm(hp_grid)
metrics_csv = open("metrics.csv", 'w')
best_result = -1
timeformat_utc = "{:%Y%m%d_%H%M%S}".format(datetime.utcnow())
train_folder = "{}_{}_{}".format(model_name, framework, timeformat_utc)
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
cmd = build_subprocess_cmd(csv_path, train_folder)


for params in pbar:
    # Generate model files according to the given hyperparameters.
    render_model(params, model_name,
                 framework, env, problem_type, target_metric, target_field, train_folder, input_types)

    # Execute model training using the generated files.
    train_generated_model(cmd, num_epochs)

    # Load the training results from the generated CSV,
    # and append to the metrics CSV.
    results = pd.read_csv("{}/metadata/results.csv".format(train_folder))
    results['trial_id'] = uuid.uuid4().hex

    results.to_csv("metrics_csv", mode="a")

    # If the target metric improves, save the new hps/files,
    # update the hyperparameters in console,
    # and delete the previous best files.

    if direction == 'max':
        top_result = results[target_metric].max()
    else:
        top_result = results[target_metric].min()

    if best_result == -1:   # if first iteration
        best_result = top_result
        shutil.copytree(train_folder, "{}_best".format(train_folder))
    else:
        is_imp = best_result > top_result
        is_imp = not is_imp if direction == 'min' else is_imp
        if is_imp:
            shutil.copytree(train_folder, "{}_best".format(train_folder))

    # Clean up the generated file folder for the next trial.
    shutil.rmtree(train_folder)


metrics_csv.close()
pbar.close()


# If running standalone, return results to parent function.


def create_model_script(csv_path, target_field, problem_type,
                        hps,
                        framework='tensorflow',
                        context='standalone'):
    """Renders the model scripts from Jinja templates.
    """

# Render pipeline.py.


pipeline = fix_code(pipeline)

# Render base.py.

base = fix_code(base)
