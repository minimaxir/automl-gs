input_dayofweeks = Input(shape=(7,), name='input_dayofweeks')
input_hours = Input(shape=(24,), name='input_hours')
{% if params['datetime_month'] %}
input_month = Input(shape=(24,), name='input_month')
{% endif %}