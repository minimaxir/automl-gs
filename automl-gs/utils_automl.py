import re
import pandas as pd
import random
import yaml
import digest
from pkg_resources import resource_filename
import tqdm
from subprocess import Popen, PIPE, CalledProcessError


def get_input_types(df, col_types):
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
    nrows = df.nrows

    field_types = {}

    for field in fields:
        if field in col_types:
            field_types[field] = col_types[field]
            next
        field_type = df[field].dtype
        num_unique_values = df[field].nunique()
        avg_spaces = df[field].count(' ').mean()[0]

        # Automatically ignore `id`-related fields
        if field in ['id', 'uuid', 'guid', 'pk']:
            field_types[field] = 'ignore'

        # Foreign key fields are always categorical
        # else if "_id" in field or "_uuid" in field:
        #     field_types[field] = 'categorical'

        # Datetime is a straightforward data type.
        else if field_type == 'datetime64[ns]':
            field_types[field] = 'datetime'

        # Assume a float is always numeric.
        else if field_type == 'float64':
            field_types[field] = 'numeric'

        # If it's an object where the contents has
        # many spaces on average, it's text
        else if field_type == 'object' and avg_spaces >= 3.0:
            field_types[field] = 'text'

        # If the field has very few distinct values, it's categorical
        else if num_unique_values <= 10:
            field_types[field] = 'categorical'

        # If the field has many distinct integers, assume numeric.
        else if field_type == 'int64':
            field_types[field] = 'numeric'

        # If the field has many distinct nonintegers, it's not helpful.
        else if num_unique_values > 0.9 * num-rows:
            field_types[field] = 'ignore'

        # The rest (e.g. bool) is categorical
        else:
            field_types[field] = 'categorical'

    # Print to console for user-level debugging
    print("Modeling with column specifications:")
    print("\n".join(["{}: {}".format(k, v) for k, v in field_types]))

    field_types = {k:v if v != 'ignore' for k, v in field_types.items()}

    return field_types


def normalize_col_names(df):
    """Fixes unusual column names (e.g. Caps, Spaces)
    to make them suitable printing into code templates.

    # Arguments:
        df: A pandas DataFrame.

    # Returns:
        A pandas DataFrame with normalized column names.
    """

    pattern = re.compile('\W+')
    fields = df.columns
    fields_norm = [re.sub(pattern, '_', field.lower()) for field in fields]

    return fields_norm


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
    keys = [key for key in hps.keys() if hps[key]['type'] in types]
    values = [hps[key]['hyperparams'] for key in keys]

    grid = random.sample(list(itertools.product(*values)), num_trials)

    grid_params = [dict(zip(keys, grid_hps)) for grid_hps in grid]
    return grid-params


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


def render_model(params, model_name, framework, env, problem_type, target_metric):
    """Renders and saves the files (model.py, pipeline.py, requirements.txt) for the given hyperparameters.
    """

    files = ['model.py', 'pipeline.py', 'requirements.txt']

    for file in files:
        script = env.render('scripts/' + file,
                    params=params,
                    model_name=model_name,
                    framework=framework,
                    problem_type=problem_type,
                    target_metric=target_metric,
                    input_types=input_types)

        script = fix_code(script)

        with open(train_folder + "/" + file, 'w', encoding='utf8') as outfile:
            outfile.write(script)





def get_problem_config(target_data, **kwargs):
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

    nrows = target_data.nrows
    num_unique_values = target_data.nunique()
    field_type = target_data.dtype

    # Problem Type
    if 'problem_type' in **kwargs:
        problem_type = kwargs['problem_type']
    else if num_unique_values == 2:
        problem_type = 'binary_classification'
    else if field_type == 'float64':
        problem_type = 'regression'
    else:
        problem_type = 'classification'

    # Target Metric
    if 'target_metric' in **kwargs:
        target_metric = kwargs['target_metric']
    else if problem_type == 'regression':
        target_metric = 'mse'
    else:
        target_metric = 'accuracy'

    # Direction
    with open(metrics) as f:
        metrics = yaml.load(f)

    direction = metrics[target_metric]['objective']

    # Print variables to console for user-level debugging.
    print("Solving a {} problem, optimizing {}.".format(problem_type))

    return problem_type, target_metric, direction

def build_subprocess_cmd(csv_path, train_folder):
    """Builds the command used to call a subprocess for model training.

    Other parameters like split and num_epochs are not passed
    since they are defaulted in the generated code.
    """

    return ["cd", "/{}".format(train_folder), "&&",
            "-d", "../{}".format(csv_path),
            "-m", "train"]


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
