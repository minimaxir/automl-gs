    # Base TensorFlow model encoding for text
    # Each text input uses the same, shared model.
    embeddings_text = Embedding(min(10000, len(encoders['tokenizer'].vocab)) + 1, {{ params['text_embed_size'] }}, name="embeddings_text")
    dropout_text = SpatialDropout1D({{ params['text_dropout'] }}, name="dropout_text")(embeddings_text)

    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        text_rnn = CuDNN{{ params['text_rnn_type'] }}({{ params['text_rnn_size'] }}, name="rnn_text")(dropout_text)
    else:
        text_rnn = {{ params['text_rnn_type'] }}({{ params['text_rnn_size'] }}, name="rnn_text",
                                recurrent_activation="sigmoid")(dropout_text)

    {% for field, field_raw, field_type in text_fields %}
    # {{ field_raw }}
    input_{{ field }} = Input(shape=({{ params['text_max_length'] }},), name="input_{{ field }}")
    embeddings_{{ field }} = embeddings_text(input_{{ field }})
    dropout_{{ field }} = dropout_text(embeddings_{{ field }})
    {{ field }}_enc = text_rnn(dropout_{{ field }})

    {% endfor %}
