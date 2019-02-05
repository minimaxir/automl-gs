{% if params['numeric_strat'] in ['minmax', 'standard'] %}
{{ field }}_enc = encoders['{{ field }}_scaler'].transform(df['{{ field }}'].values)
{% endif %}

{% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
{{ field }}_enc = pd.cut(df['{{ field }}'].values, encoders['{{ field }}_bins'], labels=False)
{{ field }}_enc = encoders['{{ field }}_scaler'].transform({{ field }}_enc)
{% endif %}