{{ field }}_dayofweeks = pd.to_datetime(df['{{ field }}']).dt.dayofweek
dayofweeks_encoder = LabelBinarizer()
dayofweeks_encoder.classes_ = list(range(7))
{{ field }}_dayofweeks = dayofweeks_encoder.transform({{ field }}_dayofweeks)

{{ field }}_hour = pd.to_datetime(df['{{ field }}']).dt.hour
hour_encoder = LabelBinarizer()
hour_encoder.classes_ = list(range(24))
{{ field }}_hour = hour_encoder.transform({{ field }}_hour)

{% if params['datetime_month'] %}
{{ field }}_month = pd.to_datetime(df['{{ field }}']).dt.month - 1
month_encoder = LabelBinarizer()
month_encoder.classes_ = list(range(12))
{{ field }}_month = month_encoder.transform({{ field }}_month)
{% endif %}

{% if params['datetime_year'] %}
{{ field }}_year = pd.to_datetime(df['{{ field }}']).dt.year
{{ field }}_year_encoder = LabelBinarizer()
{{ field }}_year = {{ field }}_year_encoder.fit_transform({{ field }}_year)
{% endif % }

{% if params['datetime_holiday'] %}
year_range = pd.to_datetime(df['{{ field }}']).dt.year
us_holidays = holidays.US(years=range(year_range.min()[0],
                                                year_range.max()[0]))
df_holidays = pd.DataFrame(list(us_holidays.items()),
                            columns=['holiday_date', 'holiday'])
holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])
holidays_df['holiday'] = holidays_df['holiday'].str.replace(' (Observed)', '')
{{ field }}_date = pd.DataFrame(pd.to_datetime(df['{{ field }}']).dt.date, columns=['date'])
{{ field }}_date = pd.merge(holidays_df, {{ field }}_date, how='right', on=['holiday_date', 'date'])

{{ field }}_holiday_encoder = LabelBinarizer()
holiday_encoded = holidays.US(years=2019)
holiday_values = [holiday.replace(" (Observed", "") for
                  holiday in holiday_encoded.values()]
{{ field }}_holiday_encoder.fit(holiday_values)

{{ field }}_holiday = {{ field }}_holiday_encoder.fit_transform({{ field }}_date['holiday'].values)

{% endif %}
