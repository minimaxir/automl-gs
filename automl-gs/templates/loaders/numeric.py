{% if params['numeric_strat'] == 'minmax' %}
{{ field }}_scaler = MinMaxScaler()
{% endif % }

{% if params['numeric_strat'] == 'standard' %}
{{ field }}_scaler = StandardScaler()
{% endif %}

with open('{{ field }}_scale.json', 'r', encoding='utf8', errors='ignore') as infile:
    {{ field }}_scaler._attrs = json.load(infile)
encoders['{{ field }}_scaler'] = {{ field }}_scaler
