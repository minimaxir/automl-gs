    # Target Field
    {% if problem_type != 'regression' %}
    {% if problem_type == 'classification' %}
    {{ target_field }}_encoder = LabelBinarizer()
    {% else %}
    {{ target_field }}_encoder = LabelEncoder()
    {% endif %}

    with open('encoders/{{ target_field }}_encoder.json', 'r', encoding='utf8', errors='ignore') as infile:
        {{ target_field }}_encoder._attrs = json.load(infile)
    encoders['{{ target_field }}_encoder'] = {{ target_field }}_encoder
    {% endif %}