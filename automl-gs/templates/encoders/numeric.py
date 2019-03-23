    # {{ field_raw }}
    {{ field }}_enc = df['{{ field_raw }}']
    {% if params['numeric_strat'] == 'minmax' %}
    {{ field }}_encoder = MinMaxScaler()
    {{ field }}_encoder_attrs = ['min_', 'scale_']
    {% endif %}
    {% if params['numeric_strat'] == 'standard' %}
    {{ field }}_encoder = StandardScaler()
    {{ field }}_encoder_attrs = ['mean_', 'var_', 'scale_']
    {% endif %}
    {% if params['numeric_strat'] == 'quantiles' %}
    {{ field }}_bins = {{ field }}_enc.quantile(np.linspace(0, 1, 4+1))
    {% endif %}
    {% if params['numeric_strat'] == 'percentiles' %}
    {{ field }}_bins = {{ field }}_enc.quantile(np.linspace(0, 1, 10+1))
    {% endif %}
    {% if params['numeric_strat'] in ['minmax', 'standard'] %}
    {{ field }}_encoder.fit(df['{{ field_raw }}'].values.reshape(-1, 1))

    {{ field }}_encoder_dict = {attr: getattr({{ field }}_encoder, attr).tolist()
                                for attr in {{ field }}_encoder_attrs}

    with open(os.path.join('encoders', '{{ field }}_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump({{ field }}_encoder_dict, outfile, ensure_ascii=False)
    {% endif %}

    {% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
    with open(os.path.join('encoders', '{{ field }}_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump({{ field }}_bins.tolist(), outfile, ensure_ascii=False)
    {% endif %}
    