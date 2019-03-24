    # {{ field_raw }}
    dayofweeks_encoder = LabelBinarizer()
    dayofweeks_encoder.classes_ = list(range(7))
    encoders['dayofweeks_encoder'] = dayofweeks_encoder

    hour_encoder = LabelBinarizer()
    hour_encoder.classes_ = list(range(24))
    encoders['hour_encoder'] = hour_encoder

    {% if params['datetime_month'] %}
    month_encoder = LabelBinarizer()
    month_encoder.classes_ = list(range(12))
    encoders['month_encoder'] = month_encoder
    {% endif %}

    {% if params['datetime_year'] %}

    {{ field }}_year_encoder = LabelBinarizer()
    with open(os.path.join('encoders', '{{ field }}_year_encoder.json'),
            'r', encoding='utf8', errors='ignore') as infile:
        {{ field }}_year_encoder.classes_ = json.load(infile)
    encoders['{{ field }}_year_encoder'] = {{ field }}_year_encoder
    {% endif %}

    {% if params['datetime_holiday'] %}
    {{ field }}_holiday_encoder = LabelBinarizer()
    holiday_encoded = holidays.US(years=2019)
    holiday_values = [holiday.replace(" (Observed", "") for
                    holiday in holiday_encoded.values()]
    {{ field }}_holiday_encoder.fit(holiday_values)
    encoders['{{ field }}_holiday_encoder'] = {{ field }}_holiday_encoder
    {% endif %}
