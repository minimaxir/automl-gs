{{ field }}_dayofweeks = pd.to_datetime(df['{{ field }}']).dt.dayofweek
dayofweeks_lb = LabelBinarizer()
dayofweeks_lb.classes_ = list(range(7))
encoders['dayofweeks_lb'] = dayofweeks_lb

{{ field }}_hour = pd.to_datetime(df['{{ field }}']).dt.hour
hour_lb = LabelBinarizer()
hour_lb.classes_ = list(range(24))
encoders['hour_lb'] = hour_lb

{% if params['datetime_month'] %}
{{ field }}_month = pd.to_datetime(df['{{ field }}']).dt.month - 1
month_lb = LabelBinarizer()
month_lb.classes_ = list(range(12))
encoders['month_lb'] = month_lb
{% endif %}

{% if params['datetime_year'] %}
{{ field }}_year = pd.to_datetime(df['{{ field }}']).dt.year
{{ field }}_year_lb = LabelBinarizer()
encoders['{{ field }}_year_lb'] = {{ field }}_year_lb
{% endif % }

{% if params['datetime_holiday'] %}
{{ field }}_holiday_lb = LabelBinarizer()
holiday_encoded = holidays.US(years=2019)
holiday_values = [holiday.replace(" (Observed", "") for
                  holiday in holiday_encoded.values()]
{{ field }}_holiday_lb.fit(holiday_values)
encoders['{{ field }}_holiday_lb'] = {{ field }}_holiday_lb
{% endif %}
