    # {{ field_raw }}
    {% if params['categorical_strat'] != 'all_binary' %}
    {{ field }}_counts = df['{{ field_raw }}'].value_counts()
    {% if params['categorical_strat'] == 'top10_perc' %}
    {{ field }}_perc = max(floor(0.1 * {{ field }}_counts.size), 1)
    {% endif %}
    {% if params['categorical_strat'] == 'top50_perc' %}
    {{ field }}_perc = max(floor(0.5 * {{ field }}_counts.size), 1)
    {% endif %}
    {{ field }}_top = np.array({{ field }}_counts.index[0:{{ field }}_perc], dtype=object)
    {{ field }}_encoder = LabelBinarizer()
    {{ field }}_encoder.fit({{ field }}_top)
    {% endif %}
    {% if params['categorical_strat'] == 'all_binary' %}
    {{ field }}_tf = df['{{ field_raw }}'].values
    {{ field }}_encoder = LabelBinarizer()
    {{ field }}_encoder.fit({{ field }}_tf)
    {% endif %}

    with open(os.path.join('encoders', '{{ field }}_encoder.json'),
             'w', encoding='utf8') as outfile:
        json.dump({{ field }}_encoder.classes_.tolist(), outfile, ensure_ascii=False)
    

