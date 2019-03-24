    # Transform and pad all text fields.

    {% for field, field_raw, _ in text_fields %}
    # {{ field_raw }}
    {{ field }}_enc = encoders['tokenizer'].transform(df['{{ field_raw }}'].values).toarray()
    
    {% endfor %}