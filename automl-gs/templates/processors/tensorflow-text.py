    # Transform and pad all text fields.

    {% for field, field_raw, _ in text_fields %}
    # {{ field_raw }}
    {{ field }}_enc = encoders['tokenizer'].texts_to_sequences(df['{{ field_raw }}'].values)
    {{ field }}_enc = pad_sequences({{ field }}_enc, maxlen={{ params['text_max_length'] }})
    
    {% endfor %}