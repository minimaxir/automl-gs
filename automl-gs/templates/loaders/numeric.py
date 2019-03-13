    # {{ field_raw }}
    {% if params['numeric_strat'] == 'minmax' %}
    {{ field }}_encoder = MinMaxScaler()
    {{ field }}_encoder_attrs = ['min_', 'scale_']
    {% endif %}
    {% if params['numeric_strat'] == 'standard' %}
    {{ field }}_encoder = StandardScaler()
    {{ field }}_encoder_attrs = ['mean_', 'var_', 'scale_']
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
    with open(os.path.join('encoders', '{{ field }}_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        {{ field }}_attrs = json.load(infile)

    for attr, value in {{ field }}_attrs.items():
        setattr({{ field }}_encoder, attr, value)
    encoders['{{ field }}_encoder'] = {{ field }}_encoder
    {% endif %}
    {% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
    with open(os.path.join('encoders', '{{ field }}_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        {{ field }}_bins = json.load(infile)
    encoders['{{ field }}_bins'] = {{ field }}_bins
    encoders['{{ field }}_encoder'] = {{ field }}_encoder
    {% endif %}