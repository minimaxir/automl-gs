import os
import pandas as pd
from jinja2 import Environment, PackageLoader
from tqdm import tqdm, tqdm_notebook
from datetime import datetime
import shutil
import uuid
import argparse
from .utils_automl import *


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
    hp_grid = build_hp_grid(framework, set(input_types.values()), num_trials, problem_type)
    fields = normalize_col_names(input_types)

    metrics_csv = open("automl_results.csv", 'w')
    best_result = None
    timeformat_utc = "{:%Y%m%d_%H%M%S}".format(datetime.utcnow())
    best_folder = "{}_{}_{}".format(model_name, framework, timeformat_utc)
    train_folder = "{}_train".format(model_name)
    cmd = build_subprocess_cmd(csv_path, train_folder)


    
    # https://stackoverflow.com/a/39662359
    try:
        is_notebook = get_ipython().__class__.__name__ in ['ZMQInteractiveShell',
                                                           'Shell']
    except:
        is_notebook = False

    pbar_func = tqdm_notebook if is_notebook else tqdm
    pbar = pbar_func(hp_grid, smoothing=0, unit='trial')
    pbar_sub = pbar_func(total=num_epochs, leave=False,
                         smoothing=0, unit='epoch')
    
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
        train_generated_model(cmd, num_epochs, train_folder, pbar_sub)

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

        top_result = train_results[target_metric]

        if top_result is not None:
            if best_result is None:   # if first iteration
                best_result = top_result
                shutil.copytree(train_folder, best_folder)
                print_progress_tqdm(params, train_results,
                                    pbar, is_notebook, False)
            else:
                is_imp = top_result > best_result
                is_imp = not is_imp if direction == 'min' else is_imp
                if is_imp:
                    best_result = top_result
                    shutil.rmtree(best_folder)
                    shutil.copytree(train_folder, best_folder)
                    print_progress_tqdm(params, train_results,
                                        pbar, is_notebook)

        # Clean up the generated file folder for the next trial.
        shutil.rmtree(train_folder)

    metrics_csv.close()
    pbar.close()
    pbar_sub.close()

def cmd():
    """Function called when invoking from the terminal."""

    parser = argparse.ArgumentParser(
        description="Provide an input CSV and a target field to predict, generate a model + code to run it. (https://github.com/minimaxir/automl-gs)"
    )



    # Explicit arguments
    parser.add_argument(
        '--csv_path',  help='Path to the CSV file (must be in the current directory) [Required]', nargs='?')
    parser.add_argument(
        '--target_field',  help="Target field to predict [Required]",
        nargs='?')
    parser.add_argument(
        '--target_metric',  help='Target metric to optimize [Default: Automatically determined depending on problem type]', nargs='?', default=None)
    parser.add_argument(
        '--framework',  help='Machine learning framework to use [Default: tensorflow]', nargs='?', default='tensorflow')
    parser.add_argument(
        '--model_name',  help=" Name of the model (if you want to train models with different names) [Default: 'automl']",
        nargs='?', default='automl')
    parser.add_argument(
        '--num_trials',  help='Number of trials / different hyperameter combos to test. [Default: 100]', nargs='?', type=int, default=100)
    parser.add_argument(
        '--split',  help="Train-val split when training the models [Default: 0.7]",
        nargs='?', type=float, default=0.7)
    parser.add_argument(
        '--num_epochs',  help='Number of epochs / passes through the data when training the models. [Default: 20]', type=int, default=20)
    parser.add_argument(
        '--gpu',  help="For non-Tensorflow frameworks and Pascal-or-later GPUs, boolean to determine whether to use GPU-optimized training methods (TensorFlow can detect it automatically) [Default: False]",
        nargs='?', type=bool, default=False)
    parser.add_argument(
        '--tpu_address',  help="For TensorFlow, hardware address of the TPU on the system. [Default: None]",
        nargs='?', default=None)

    # Positional arguments
    parser.add_argument('csv_path', nargs='?')
    parser.add_argument('target_field', nargs='?')

    args = parser.parse_args()
    automl_grid_search(csv_path=args.csv_path,
                       target_field=args.target_field,
                       target_metric=args.target_metric,
                       framework=args.framework,
                       model_name=args.model_name,
                       num_trials=args.num_trials,
                       split=args.split,
                       num_epochs=args.num_epochs,
                       gpu=args.gpu,
                       tpu_address=args.tpu_address)
