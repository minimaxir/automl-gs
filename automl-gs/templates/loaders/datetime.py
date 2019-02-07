{{ field }}_dayofweeks = pd.to_datetime(df['{{ field }}']).dt.dayofweek
dayofweeks_encoder = LabelBinarizer()
dayofweeks_encoder.classes_ = list(range(7))
encoders['dayofweeks_encoder'] = dayofweeks_encoder

{{ field }}_hour = pd.to_datetime(df['{{ field }}']).dt.hour
hour_encoder = LabelBinarizer()
hour_encoder.classes_ = list(range(24))
encoders['hour_encoder'] = hour_encoder

{% if params['datetime_month'] %}
{{ field }}_month = pd.to_datetime(df['{{ field }}']).dt.month - 1
month_encoder = LabelBinarizer()
month_encoder.classes_ = list(range(12))
encoders['month_encoder'] = month_encoder
{% endif %}

{% if params['datetime_year'] %}
{{ field }}_year = pd.to_datetime(df['{{ field }}']).dt.year
{{ field }}_year_encoder = LabelBinarizer()
encoders['{{ field }}_year_encoder'] = {{ field }}_year_encoder
{% endif % }

{% if params['datetime_holiday'] %}
{{ field }}_holiday_encoder = LabelBinarizer()
holiday_encoded = holidays.US(years=2019)
holiday_values = [holiday.replace(" (Observed", "") for
                  holiday in holiday_encoded.values()]
{{ field }}_holiday_encoder.fit(holiday_values)
encoders['{{ field }}_holiday_encoder'] = {{ field }}_holiday_encoder
{% endif %}
