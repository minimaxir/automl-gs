    {% if problem_type != 'regression' %}
    {{ target_field }}_enc = df['{{ target_field }}'].values

    {{ target_field }}_enc = encoders['{{ target_field }}_encoder'].transform({{ target_field }}_enc)
    {% endif %}