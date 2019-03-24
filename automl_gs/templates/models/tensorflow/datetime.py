    # {{ field_raw }}
    input_dayofweeks_{{ field }} = Input(shape=(7,), name="input_dayofweeks_{{ field }}")
    input_hours_{{ field }} = Input(shape=(24,), name="input_hours_{{ field }}")
    {% if params['datetime_month'] %}
    input_month_{{ field }} = Input(shape=(12,), name="input_month_{{ field }}")
    {% endif %}
    {% if params['datetime_year'] %}
    input_year_{{ field }}_size = len(encoders['{{ field }}_year_encoder'].classes_)
    input_year_{{ field }} = Input(shape=(input_year_{{ field }}_size if input_year_{{ field }}_size != 2 else 1,), name="input_year_{{ field }}")
    {% endif %}