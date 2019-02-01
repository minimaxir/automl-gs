from tf.contrib.opt import AdamWOptimizer
from tf.train import cosine_decay

def build_model():
    """Builds and compiles the model from scratch.

    # Returns
        model: A compiled model which can be used to train or predict.
    """
    {% if 'text' in params.values() %}
    {% include 'models/' ~ framework ~ '/text.py' %}
    {% endif %}

    {% for field, field_type in params %}
        {% if field_type != 'text' %}
        {% include 'models/' ~ framework ~ '/' ~ field_type ~ '.py' %}
        {% endif %}
    {% endfor %}

    concat = concatenate([
        {% for field, field_type in params %}
        {{ field }}_enc{{ ", " if not loop.last }}
        {% endfor %}
        ], name='concat')

    {% include 'models/' ~ framework ~ '/mlp.py' %}

    model = Model(inputs=[
        {% for field, field_type in params %}
        input{{ field }}{{ ", " if not loop.last }}
        {% endfor %}
                ],
                      outputs=[output])

    global_step = tf.Variable(0, trainable=False)
    lr_decayed = cosine_decay({{ learning_rate }}, global_step, 1000)
    model.compile(loss=hybrid_loss,
              optimizer=AdamWOptimizer(learning_rate = lr_decayed,
                                        weight_decay = {{ weight_decay }})

    return model


def build_encoders(df):
    """Builds encoders for necessary fields to be used when
    transforming data for the model.

    All encoder specifications are stored locally as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """
    {% if 'text' in params.values() %}
    {% include 'encoders/' ~ framework ~ '-text.py' %}
    {% endif %}

    {% for field, field_type in params %}
        {% if field_type != 'text' %}
        {% include 'encoders/' ~ field_type ~ '.py' %}
        {% endif %}
    {% endfor %}

def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects.
    """

def process_data(df):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a pandas DataFrame containing the source data

    # Returns
        A list containing all the processed fields to be fed
        into the model.
    """

    {% if 'text' in params.values() %}
    {% include 'processors/' ~ framework ~ '-text.py' %}
    {% endif %}

    {% for field, field_type in params %}
        {% if field_type != 'text' %}
        {% include 'processors/' ~ field_type ~ '.py' %}
        {% endif %}
    {% endfor %}

    return [{% for field, field_type in params %}
        {{ field }}_enc{{ ", " if not loop.last }}
        {% endfor %}]


def model_predict(df, model):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df)
    {% if framework == 'tensorflow' %}
    return model.predict(data_enc)
    {% endif %}

def model_train(df, model):
    """Trains a model, and saves the data locally.
    Also rebuilds the encoders to fit the new data.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
    """
    build_encoders(df)
    data_enc = process_data(df)

    target = df['{{ target_metric }}'].values

    meta_callback = meta_callback()

def meta_callback():
    """Callback used during model training to save current weights and logs after each training epoch.

    """

    # Only run while using automl-gs, which tells it an epoch is finished.

