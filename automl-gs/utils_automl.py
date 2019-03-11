import re
import pandas as pd
import random
import yaml
import os
from pkg_resources import resource_filename
import tqdm
from subprocess import Popen, PIPE, CalledProcessError
from autopep8 import fix_code
from collections import OrderedDict


def get_input_types(df, col_types, target_field):
    """Get the input types for each field in the DataFrame that corresponds
    to an input type to be fed into the model.

    Valid values are ['text', 'categorical', 'numeric', 'datetime', 'ignore']

    # Arguments:
        df: A pandas DataFrame.
        col_types: A dict of explicitly defined {field_name: type} mappings.

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
            next
        field_type = df[field].dtype
        num_unique_values = df[field].nunique()
        if field_type == 'object':
            avg_spaces = df[field].str.count(' ').mean()

        # Automatically ignore `id`-related fields
        if field in ['id', 'uuid', 'guid', 'pk']:
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
                  hp_path=resource_filename(__name__, "hyperparameters.yml")):
    """Builds the hyperparameter grid for model grid search.

    # Arguments:
        framework: string indicating the framework (e.g. `tensorflow`)
        types: list of hyperparameter types to consider; exclude rest
        num_trials: number of distinct trials to keep
        hp_path: filepath of hyperparameters

    # Returns
        A list of dicts of hyperparameter specifications
    """

    with open(hp_path) as f:
        hps = yaml.load(f)

    # Refine hyperparameters by only using ones relevant to
    # the data and framework of choice
    hps = dict(hps['base'], **hps[framework])
    keys = [key for key in hps.keys() if (hps[key]['type'] in types
                                          or hps[key]['type'] == 'base')]
    values = [hps[key]['hyperparams'] for key in keys]

    grid = set()
    while len(grid) < num_trials:
        grid.add(tuple([random.choice(x) for x in values]))

    grid_params = [dict(zip(keys, grid_hps)) for grid_hps in grid]
    return grid_params


def print_progress_tqdm(hps, metrics):
    """Custom writer for tqdm which prints winning metrics and hyperparameters
    to console after each iteration.

    Uses a hack for tqdm.write(): https://github.com/tqdm/tqdm/issues/520

    # Arguments:
        hps: dict of hyperparameters
        metrics: dict of metrics
    """

    hp_str = '\n'.join(['{}: {}'.format(k, v) for k, v in hp.items()])
    metrics_str = '\n'.join(['{}: {}'.format(k, v)
                             for k, v in metrics.items()])

    console_str = ("Metrics:\n" + hp_str + "\n" +
                   "Hyperparameters:\n" + metrics_str)

    # Print to console, removing appropriate number of lines
    tqdm.write([_term_move_up()] *
               console_str.count("\n") + '/r' + console_str)


def render_model(params, model_name, framework, env, problem_type, 
                 target_metric, target_field, train_folder, fields, split, num_epochs,
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
    nontarget_fields = [field for field in fields if field[0] != target_field]
    has_text_input = 'text' in [field[2] for field in fields]

    with open(metrics_path) as f:
        metrics = yaml.load(f)[problem_type]

    for file in files:
        script = env.get_template('scripts/' + file).render(
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
            has_text_input=has_text_input,
            metrics=metrics)

        script = fix_code(script)

        with open(train_folder + "/" + file, 'w', encoding='utf8') as outfile:
            outfile.write(script)


def get_problem_config(target_data,
                       framework,
                       metrics_path=resource_filename(__name__, "metrics.yml"),
                       **kwargs):
    """Gets the problem type, target metric, and metric direction, or infers
    them from the data if not expicitly specified.

    # Arguments:
        target_data: Data column to infer problem spec on.

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
    if 'problem_type' in kwargs:
        problem_type = kwargs['problem_type']
    elif num_unique_values == 2:
        problem_type = 'binary_classification'
    elif field_type == 'float64':
        problem_type = 'regression'
    else:
        problem_type = 'classification'

    # Target Metric
    if 'target_metric' in kwargs:
        target_metric = kwargs['target_metric']
    elif problem_type == 'regression':
        target_metric = 'mse'
    else:
        target_metric = 'accuracy'

    # Direction
    with open(metrics_path) as f:
        metrics = yaml.load(f)

    direction = metrics[target_metric]['objective']
    direction_text = 'minimizing' if direction == 'min' else 'maximizing'

    # Print config to console for user-level debugging.
    print("Solving a {} problem, {} {} using {}.".format(
        problem_type, direction_text, target_metric, framework))

    return problem_type, target_metric, direction


def build_subprocess_cmd(csv_path, train_folder):
    """Builds the command used to call a subprocess for model training.

    Other parameters like split and num_epochs are not passed
    since they are the default in the generated code.
    """

    csv_path_join = os.path.join('..', csv_path)

    return ["cd", train_folder, "&&",
            "python3", "model.py",
            "-d", csv_path_join,
            "-m", "train",
            "-c", "automl-gs"]


def train_generated_model(cmd, num_epochs):
    """Trains a generated model script in a Python subprocess,
       and maintains a progress bar of the subprocess training.

       Each subprocess must output a stdout flush + an
       "EPOCH_END" string accordingly

    # Arguments:
        cmd: A generate command
    """

    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        with tqdm(total=num_epochs) as t:
            for line in p.stdout:
                if line == "EPOCH_END\n":
                    t.update(1)
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)
