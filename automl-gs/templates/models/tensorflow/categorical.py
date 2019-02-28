    # {{ field_raw }}
    input_{{ field }} = Input(shape=(encoders['{{ field }}_encoder'].classes_.shape[0],), name="input_{{ field }}")

