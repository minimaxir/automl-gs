    # {{ field_raw }}
    {{ field }}_enc = df['{{ field_raw }}'].values
    {{ field }}_enc = encoders['{{ field }}_encoder'].transform({{ field }}_enc)
    