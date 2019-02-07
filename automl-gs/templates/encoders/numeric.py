{{ field }}_enc = df['{{ field }}'].values

{% if params['numeric_strat'] == 'minmax' %}
{{ field }}_encoder = MinMaxScaler()
{% endif % }

{% if params['numeric_strat'] == 'standard' %}
{{ field }}_encoder = StandardScaler()
{% endif %}

{% if params['numeric_strat'] == 'quantiles' %}
{{ field }}_bins = np.percentile({{ field }}_enc, range(25, 75, 25))
{% endif %}

{% if params['numeric_strat'] == 'percentiles' %}
{{ field }}_bins = np.percentile({{ field }}_enc, range(10, 90, 10))
{% endif %}

{% if params['numeric_strat'] in ['minmax', 'standard'] %}
{{ field }_encoder.fit(df['{{ field }}'].values)

with open('{{ field }}_encoder.json', 'w', encoding='utf8') as outfile:
    json.dump({{ field }}_encoder._attrs, outfile, ensure_ascii=False)
{% endif %}

{% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
with open('{{ field }}_bins.json', 'w', encoding='utf8') as outfile:
    json.dump({{ field }}_bins, outfile, ensure_ascii=False)
{% endif %}