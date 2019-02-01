{% extends "tensorflow-mlp.py" %}
{% block mlp %}
hidden = Dense({{ mlp_first_size }}, activation='relu', name='hidden_1', kernel_regularizer={{ mlp_regularizer }})(concat)
hidden = BatchNormalization(name="bn_1")(hidden)
hidden = Dropout({{ mlp_dropout }}, name="dropout_1")(hidden)

for i in range(mlp_blocks-1):
    hidden = Dense({{ mlp_size }}, activation='relu', name='hidden_{}'.format(i+2), kernel_regularizer={{ mlp_regularizer }})(hidden)
    hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
    hidden = Dropout({{ mlp_dropout }}, name="dropout_{}".format(i + 2))(hidden)
{% endblock %}