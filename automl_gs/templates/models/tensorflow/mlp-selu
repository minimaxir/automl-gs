    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense({{ params['mlp_first_size'] }}, activation='selu', name='hidden_1', kernel_regularizer={{ params['mlp_regularizer'] }})(concat)
    hidden = AlphaDropout({{ params['mlp_dropout'] }}, name="dropout_1")(hidden)

    for i in range({{ params['mlp_blocks'] }}-1):
        hidden = Dense({{ params['mlp_size'] }}, activation="selu", name="hidden_{}".format(i+2), kernel_regularizer={{ params['mlp_regularizer'] }})(hidden)
        hidden = AlphaDropout({{ params['mlp_dropout'] }}, name="dropout_{}".format(i+2))(hidden)