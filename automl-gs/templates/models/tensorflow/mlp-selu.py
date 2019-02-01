{% extends "tensorflow-mlp.py" %}
{% block mlp %}
hidden = Dense({{ mlp_first_size }}, activation='selu', name='hidden_1', kernel_regularizer={{ mlp_regularizer }})(concat)
hidden = AlphaDropout({{ mlp_dropout }}, name="dropout_1")(hidden)

for i in range(mlp_blocks-1):
    hidden = Dense({{ mlp_size }}, activation='selu', name='hidden_{}'.format(i+2), kernel_regularizer={{ mlp_regularizer }})(hidden)
    hidden = AlphaDropout({{ mlp_dropout }}, name="dropout_{}".format(i + 2))(hidden)
{% endblock %}