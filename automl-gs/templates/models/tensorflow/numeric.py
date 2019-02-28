    # {{ field_raw }}
    {% if params['numeric_strat'] in ['minmax', 'standard'] %}
    input_{{ field }} = Input(shape=(1,), name="input_{{ field }}")
    {% elif params['numeric_strat'] == 'quantiles' %}
    input_{{ field }} = Input(shape=(4,), name="input_{{ field }}")
    {% else %}
    input_{{ field }} = Input(shape=(10,), name="input_{{ field }}")
    {% endif %}