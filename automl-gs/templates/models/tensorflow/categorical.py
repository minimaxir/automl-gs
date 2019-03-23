    # {{ field_raw }}
    input_{{ field }}_size = len(encoders['{{ field }}_encoder'].classes_)
    input_{{ field }} = Input(shape=(input_{{ field }}_size if input_{{ field }}_size != 2 else 1,), name="input_{{ field }}")

