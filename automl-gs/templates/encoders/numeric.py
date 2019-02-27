    # {{ field_raw }}
    {{ field }}_enc = df['{{ field_raw }}'].values
    {% if params['numeric_strat'] == 'minmax' %}
    {{ field }}_encoder = MinMaxScaler()
    {% endif %}
    {% if params['numeric_strat'] == 'standard' %}
    {{ field }}_encoder = StandardScaler()
    {% endif %}
    {% if params['numeric_strat'] == 'quantiles' %}
    {{ field }}_bins = {{ field }}_enc.quantile(np.linspace(0, 1, 4+1))
    {% endif %}
    {% if params['numeric_strat'] == 'percentiles' %}
    {{ field }}_bins = {{ field }}_enc.quantile(np.linspace(0, 1, 10+1))
    {% endif %}
    {% if params['numeric_strat'] in ['minmax', 'standard'] %}
    {{ field }}_encoder.fit(df['{{ field }}'].values)

    with open('encoders/{{ field }}_encoder.json', 'w', encoding='utf8') as outfile:
        json.dump({{ field }}_encoder._attrs, outfile, ensure_ascii=False)
    {% endif %}

    {% if params['numeric_strat'] in ['quantiles', 'percentiles'] %}
    with open('encoders/{{ field }}_bins.json', 'w', encoding='utf8') as outfile:
        json.dump({{ field }}_bins, outfile, ensure_ascii=False)
    {% endif %}
    