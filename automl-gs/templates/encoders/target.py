    {% if problem_type != 'regression' %}
    {% if problem_type == 'classification' %}
    {{ target_field }}_encoder = LabelBinarizer()
    {% else %}
    {{ target_field }}_encoder = LabelEncoder()
    {% endif %}
    {{ target_field }}_encoder.fit(df['{{ target_field }}'].values)

    with open('encoders/{{ target_field }}_encoder.json', 'w', encoding='utf8') as outfile:
        json.dump({{ target_field }}_encoder.classes_, outfile, ensure_ascii=False)
    {% endif %}