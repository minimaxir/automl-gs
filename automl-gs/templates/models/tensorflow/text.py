# Text

{% for field, field_type in params %}
    {% if field_type == 'text' %}
    input_{{ field }} = Input(shape=({{ text_max_words }},), name='input_{{ field }}')
    {% endif %}
{% endfor %}

# Base TensorFlow model encoding for text
# Each text input uses the shared model.

embeddings_text = Embedding({{ text_max_words }} + 1, 50, name='embeddings_text')
dropout_text = SpatialDropout1D({{ text_dropout }}, name='dropout_text')(embeddings_text)

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    text_rnn = CuDNN{{ text_rnn_type }}({{ text_rnn_size }}, name='rnn_text')(dropout_text)
else:
    text_rnn = {{ text_rnn_type }}({{text_rnn_size}}, name='rnn_text',
                            recurrent_activation='sigmoid')(dropout_text)

{% for field, field_type in params %}
    {% if field_type == 'text' %}
    embeddings_{{ field }} = embeddings_text(input_{{ field }})
    dropout_{{ field }} = dropout_text(embeddings_{{ field }})
    {{ field }}_enc = text_rnn(dropout_{{ field }})
    {% endif %}
{% endfor %}
