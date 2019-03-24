    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense({{ params['mlp_first_size'] }}, activation="relu", name="hidden_1", kernel_regularizer={{ params['mlp_regularizer'] }})(concat)
    hidden = BatchNormalization(name="bn_1")(hidden)
    hidden = Dropout({{ params['mlp_dropout'] }}, name="dropout_1")(hidden)

    for i in range({{ params['mlp_blocks'] }}-1):
        hidden = Dense({{ params['mlp_size'] }}, activation="relu", name="hidden_{}".format(i+2), kernel_regularizer={{ params['mlp_regularizer'] }})(hidden)
        hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
        hidden = Dropout({{ params['mlp_dropout'] }}, name="dropout_{}".format(i+2))(hidden)
