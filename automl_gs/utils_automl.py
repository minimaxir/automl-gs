import re
import pandas as pd
import random
import yaml
import os
import shutil
from time import time
from pkg_resources import resource_filename
from tqdm import tqdm, tqdm_notebook
from tqdm._utils import _term_move_up
from subprocess import Popen, PIPE, DEVNULL, CalledProcessError
from autopep8 import fix_code
from collections import OrderedDict


def get_input_types(df, col_types, target_field):
    """Get the input types for each field in the DataFrame that corresponds
    to an input type to be fed into the model.

    Valid values are ['text', 'categorical', 'numeric', 'datetime', 'ignore']

    # Arguments:
        df: A pandas DataFrame.
        col_types: A dict of explicitly defined {field_name: type} mappings.
        target_field: string indicating the target field

    # Returns:
        A dict of {field_name: type} mappings.
    """

    fields = df.columns
    nrows = df.shape[0]
    avg_spaces = -1

    field_types = OrderedDict()

    for field in fields:
        if field in col_types:
            field_types[field] = col_types[field]
            continue
        field_type = df[field].dtype
        num_unique_values = df[field].nunique()
        if field_type == 'object':
            avg_spaces = df[field].str.count(' ').mean()

        # Automatically ignore `id`-related fields
        if field.lower() in ['id', 'uuid', 'guid', 'pk', 'name']:
            field_types[field] = 'ignore'

        # Foreign key fields are always categorical
        # else if "_id" in field or "_uuid" in field:
        #     field_types[field] = 'categorical'

        # Datetime is a straightforward data type.
        elif field_type == 'datetime64[ns]':
            field_types[field] = 'datetime'

        # Assume a float is always numeric.
        elif field_type == 'float64':
            field_types[field] = 'numeric'

        # If it's an object where the contents has
        # many spaces on average, it's text
        elif field_type == 'object' and avg_spaces >= 2.0:
            field_types[field] = 'text'

        # If the field has very few distinct values, it's categorical
        elif num_unique_values <= 10:
            field_types[field] = 'categorical'

        # If the field has many distinct integers, assume numeric.
        elif field_type == 'int64':
            field_types[field] = 'numeric'

        # If the field has many distinct nonintegers, it's not helpful.
        elif num_unique_values > 0.9 * nrows:
            field_types[field] = 'ignore'

        # The rest (e.g. bool) is categorical
        else:
            field_types[field] = 'categorical'

    # Print to console for user-level debugging
    print("Modeling with field specifications:")
    print("\n".join(["{}: {}".format(k, v) for k, v in field_types.items() if k != target_field]))

    field_types = {k: v for k, v in field_types.items() if v != 'ignore'}

    return field_types


def normalize_col_names(input_types):
    """Fixes unusual column names (e.g. Caps, Spaces)
    to make them suitable printing into code templates.

    # Arguments:
        input_types: dict of col names: input types

    # Returns:
        A dict of col names: input types with normalized keys
    """

    pattern = re.compile('\W+')
    fields = [(re.sub(pattern, '_', field.lower()), field, field_type)
                   for field, field_type in input_types.items()]

    return fields


def build_hp_grid(framework, types, num_trials,
                  problem_type,
                  hp_path=resource_filename(__name__, "hyperparameters.yml")):
    """Builds the hyperparameter grid for model grid search.

    # Arguments:
        framework: string indicating the framework (e.g. `tensorflow`)
        types: list of hyperparameter types to consider; exclude rest
        num_trials: number of distinct trials to keep
        problem_type: type of problem to solve
        hp_path: filepath of hyperparameters

    # Returns
        A list of dicts of hyperparameter specifications
    """

    with open(hp_path) as f:
        hps = yaml.safe_load(f)

    # Refine hyperparameters by only using ones relevant to
    # the data and framework of choice
    hps = dict(hps['base'], **hps[framework])
    keys = [key for key in hps.keys() if (hps[key]['type'] in types
                                          or hps[key]['type'] == 'base'
                                          or hps[key]['type'] == problem_type)]
    values = [hps[key]['hyperparams'] for key in keys]

    grid = set()
    while len(grid) < num_trials:
        grid.add(tuple([random.choice(x) for x in values]))

    grid_params = [dict(zip(keys, grid_hps)) for grid_hps in grid]
    return grid_params


def print_progress_tqdm(hps, metrics, pbar, is_notebook, clear=True):
    """Custom writer for tqdm which prints winning metrics
    to console after each iteration.

    Uses a hack for tqdm.write(): https://github.com/tqdm/tqdm/issues/520

    # Arguments:
        hps: dict of hyperparameters
        metrics: dict of hyperparameters+metrics
        pbar: a tqdm progressbar
        is_notebook: boolean if automl-gs is running in a Notebook.
        clear: if writing should clear existing output
    """

    # hp_str = '\n'.join(['{}: {}'.format(k, v) for k, v in hps.items()])
    metrics_str = '\n'.join(['{}: {}'.format(k, v) for k, v in metrics.items()
                             if k not in hps.keys()])

    # console_str = ("\nHyperparameters:\n" + hp_str + "\n" +
    #              "\nMetrics:\n" + metrics_str)

    console_str = "\nMetrics:\n" + metrics_str

    # Print to console, removing appropriate number of lines
    move_up_char = '' if is_notebook else _term_move_up()
    if clear:
        pbar.write("".join([move_up_char] * (console_str.count('\n') + 2)))

    pbar.write(console_str)


def render_model(params, model_name, framework, env, problem_type, 
                 target_metric, target_field, train_folder, fields,
                 split, num_epochs, gpu, tpu_address,
                 metrics_path=resource_filename(__name__, "metrics.yml")):
    """Renders and saves the files (model.py, pipeline.py, requirements.txt) for the given hyperparameters.
    """

    files = ['model.py', 'pipeline.py', 'requirements.txt']

    type_map = {
    'numeric': 'float64',
    'categorical': 'str',
    'datetime': 'str',
    'text': 'str'
    }

    load_fields = {field[1]: type_map[field[2]] for field in fields}
    text_fields = [field for field in fields if field[2] == 'text']
    nontarget_fields = [field for field in fields if field[1] != target_field]
    target_field, target_field_raw = [(field[0], field[1]) for field in fields if field[1] == target_field][0]
    has_text_input = 'text' in [field[2] for field in fields]
    text_framework = 'tensorflow' if framework == 'tensorflow' else 'sklearn'

    with open(metrics_path) as f:
        metrics = yaml.safe_load(f)[problem_type]

    for file in files:
        script = env.get_template('scripts/' + file.replace('.py', '')).render(
            params=params,
            model_name=model_name,
            framework=framework,
            problem_type=problem_type,
            target_metric=target_metric,
            target_field=target_field,
            fields=fields,
            split=split,
            num_epochs=num_epochs,
            load_fields=load_fields,
            text_fields=text_fields,
            nontarget_fields=nontarget_fields,
            target_field_raw=target_field_raw,
            has_text_input=has_text_input,
            metrics=metrics,
            text_framework=text_framework,
            gpu=gpu,
            tpu_address=tpu_address)

        script = fix_code(script)

        with open(train_folder + "/" + file, 'w', encoding='utf8') as outfile:
            outfile.write(script)


def get_problem_config(target_data,
                       framework,
                       target_metric,
                       metrics_path=resource_filename(__name__, "metrics.yml")):
    """Gets the problem type, target metric, and metric direction, or infers
    them from the data if not expicitly specified.

    # Arguments:
        target_data: Data column to infer problem spec on.
        framework: problem framework
        target_metric: Target metric to optimize (overrides automatic selection)
        metrics_path: location of the metrics file

    # Returns:
        problem_type: One of 'regression', 'binary_classification' or
                      'classification'.
        target_metric: Target metric to optimize.
        direction: Direction of the metric to optimize (either 'max' or 'min')
    """

    nrows = target_data.size
    num_unique_values = target_data.nunique()
    field_type = target_data.dtype

    # Problem Type
    if num_unique_values == 2:
        problem_type = 'binary_classification'
    elif field_type == 'float64':
        problem_type = 'regression'
    else:
        problem_type = 'classification'

    # Target Metric
    if target_metric is not None:
        pass
    elif problem_type == 'regression':
        target_metric = 'mse'
    else:
        target_metric = 'accuracy'

    # Direction
    with open(metrics_path) as f:
        metrics = yaml.safe_load(f)

    direction = metrics[target_metric]['objective']
    direction_text = 'minimizing' if direction == 'min' else 'maximizing'

    # Print config to console for user-level debugging.
    print("Solving a {} problem, {} {} using {}.\n".format(
        problem_type, direction_text, target_metric, framework))

    return problem_type, target_metric, direction


def build_subprocess_cmd(csv_path, train_folder):
    """Builds the command used to call a subprocess for model training.

    Other parameters like split and num_epochs are not passed
    since they are the default in the generated code.
    """

    csv_path_join = os.path.join('..', csv_path)

    # Find the python executable
    if shutil.which('python3') is not None:
        pycmd = shutil.which('python3')
    elif shutil.which('python'):
        # fall back to regular python, which may be py3
        pycmd = shutil.which('python')
    else:
        # might be a better exception for this
        raise Exception("error: unable to locate the python binary for the subprocess call")

    return [pycmd, "model.py",
            "-d", csv_path_join,
            "-m", "train",
            "-c", "automl-gs"]


def train_generated_model(cmd, num_epochs, train_folder, pbar_sub):
    """Trains a generated model script in a Python subprocess,
       and maintains a progress bar of the subprocess training.

       Each subprocess must output a stdout flush + an
       "EPOCH_END" string accordingly

    # Arguments:
        cmd: A generate command
        num_epochs: number of epochs
        train_folder: subfolder where the training occurs.
        pbar_sub: tqdm progress bar for the subprocess
    """

    p = Popen(cmd, cwd=train_folder, stdout=PIPE, bufsize=1,
              universal_newlines=True) 
        
    for line in iter(p.stdout.readline, ""):
        if line == "EPOCH_END\n":
            pbar_sub.update(1)

    if p.returncode is not None:
        raise CalledProcessError(p.returncode, p.args)

    p.stdout.close()

    # Reset the subprogress bar without destroying it
    # https://github.com/tqdm/tqdm/issues/545#issuecomment-471090550
    pbar_sub.n = 0
    pbar_sub.last_print_n = 0
    pbar_sub.start_t = time()
    pbar_sub.last_print_t = time()
    pbar_sub.refresh()
