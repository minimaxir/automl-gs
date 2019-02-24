    # Transform and pad all text fields.

    {% for field, field_type in input_types.items() %}
        {% if field_type == 'text' %}
    {{ field }}_enc = encoders['tokenizer'].texts_to_sequences(df['{{ field }}'].values)
    {{ field }}_enc= sequence.pad_sequences({{ field }}_enc, maxlen={{ params['text_max_length'] }})
        {% endif %}
    {% endfor %}