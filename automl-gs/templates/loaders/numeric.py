{% if params['numeric_strat'] == 'minmax' %}
{{ field }}_scaler = MinMaxScaler()
{% endif % }

{% if params['numeric_strat'] == 'standard' %}
{{ field }}_scaler = StandardScaler()
{% endif %}

{% if params['numeric_strat'] == 'quantiles' %}
{{ field }}_encoder = LabelBinarizer()
{{ field }}_encoder.classes_ = list(range(4))
{% endif %}

{% if params['numeric_strat'] == 'percentiles' %}
{{ field }}_encoder = LabelBinarizer()
{{ field }}_encoder.classes_ = list(range(10))
{% endif %}

{% if params['numeric_strat'] in ['minmax', 'standard'] %}
with open('{{ field }}_scale.json', 'r', encoding='utf8', errors='ignore') as infile:
    {{ field }}_scaler._attrs = json.load(infile)
encoders['{{ field }}_scaler'] = {{ field }}_scaler
{% endif %}

{% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
with open('{{ field }}_scale.json', 'r', encoding='utf8', errors='ignore') as infile:
    {{ field }}_bins = json.load(infile)
encoders['{{ field }}_bins'] = {{ field }}_bins
encoders['{{ field }}_encoder'] = {{ field }}_encoder
{% endif %}