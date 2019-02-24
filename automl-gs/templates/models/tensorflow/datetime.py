input_dayofweeks_{{ field }} = Input(shape=(7,), name='input_dayofweeks_{{ field }}')
input_hours_{{ field }} = Input(shape=(24,), name='input_hours_{{ field }}')
{% if params['datetime_month'] %}
input_month_{{ field }} = Input(shape=(24,), name='input_month_{{ field }}')
{% endif %}