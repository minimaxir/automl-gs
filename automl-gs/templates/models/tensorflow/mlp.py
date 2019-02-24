    mlp_blocks = {{ params['mlp_blocks'] }}

{% include 'models/tensorflow/mlp-' ~ params['mlp_activation'] ~ '.py' %}
        
    {% if problem_type == 'regression' %}
    output = Dense(1, activation='relu', name='output', kernel_regularizer={{ params['output_regularizer'] }})(hidden)
    {% endif %}

    {% if problem_type == 'binary' %}
    output = Dense(1, activation='sigmoid', name='output', kernel_regularizer={{ params['output_regularizer'] }})(hidden)
    {% endif %}

    {% if problem_type == 'classification' %}
    output = Dense(num_classes, activation='softmax', name='output', kernel_regularizer={{ params['output_regularizer'] }})(hidden)
    {% endif %}
