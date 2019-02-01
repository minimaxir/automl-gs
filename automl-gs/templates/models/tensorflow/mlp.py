mlp_blocks = {{ mlp_blocks }}

{% include 'models/tensorflow/mlp-' ~ params['mlp_activation'] ~ '.py' %}
    
{% if params['problem_type'] == 'regression' %}
output = Dense(1, activation='relu', name='output', kernel_regularizer=output_regularizer)(hidden)
{% endif %}

{% if params['problem_type'] == 'binary' %}
output = Dense(1, activation='sigmoid', name='output', kernel_regularizer=output_regularizer)(hidden)
{% endif %}

{% if params['problem_type'] == 'classification' %}
output = Dense(num_classes, activation='softmax', name='output', kernel_regularizer=output_regularizer)(hidden)
{% endif %}

output = Dense(1, activation='relu', name='output', kernel_regularizer=output_regularizer)(hidden)
