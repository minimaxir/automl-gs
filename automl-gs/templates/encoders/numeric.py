{{ field }}_enc = df['{{ field }}'].values

{% if params['numeric_strat'] == 'minmax' %}
{{ field }}_scaler = MinMaxScaler()
{% endif % }

{% if params['numeric_strat'] == 'standard' %}
{{ field }}_scaler = StandardScaler()
{% endif %}

{{ field }_scaler.fit(df['{{ field }}'].values)

with open('{{ field }}_scale.json', 'w', encoding='utf8') as outfile:
    json.dump({{ field }}_scaler._attrs, outfile, ensure_ascii=False)