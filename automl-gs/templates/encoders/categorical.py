    {{ field }}_tf = df['{{ field }}'].values

    {% if params['categorical_strat'] != 'all_binary' %}
    {{ field }}_counts = df['{{ field }}'].value_counts()

    {% if params['categorical_strat'] != 'top10_perc' %}
    {{ field }}_perc = floor(0.1 * {{ field }}_counts.size)
    {% endif %}

    {% if params['categorical_strat'] != 'top50_perc' %}
    {{ field }}_perc = floor(0.5 * {{ field }}_counts.size)
    {% endif %}

    {{ field }}_top = np.array({{ field }}_counts.index[0:{{ field }}_perc], dtype=object)

    {{ field }}_encoder = LabelBinarizer()
    {{ field }}_encoder.fit({{ field }}_top)
    {% endif %}

    {% if params['categorical_strat'] == 'all_binary' %}
    {{ field }}_encoder = LabelBinarizer()
    {{ field }}_encoder.fit({{ field }}_tf)
    {% endif %}

    with open('encoders/{{ field }}_encoder.json', 'w', encoding='utf8') as outfile:
        json.dump({{ field }}_encoder.classes_, outfile, ensure_ascii=False)