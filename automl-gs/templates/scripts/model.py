{% include 'imports/model.py' %}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script which utilizes a model trained to predict {{ target_field }}."
                    "Script created using automl-gs (https://github.com/minimaxir/automl-gs)")
    parser.add_argument('-d', '--data',  help="Input dataset (must be a .csv)")
    parser.add_argument(
        '-m', '--mode',  help='Mode (either "train" or "predict")')
    parser.add_argument(
    '-s', '--split',  help="Train/Validation Split (if training)",
    default={{ split }})
    parser.add_argument(
    '-e', '--epochs',  help="# of Epochs (if training)",
    default={{ num_epochs }})
    parser.add_argument(
    '-c', '--context',  help="Context for running script (used during automl-gs training)",
    default='standalone')
    args = parser.parse_args()

    cols = [{% for _, raw_field, _  in fields %}
            "{{ raw_field }}"{{ ", " if not loop.last }}
            {% endfor %}]
    dtypes = {{ load_fields }}

    df = pd.read_csv(args.data, parse_dates=True,
                     usecols=cols,
                     dtype=dtypes)

    if args.mode == 'train':
        build_encoders(df)
        encoders = load_encoders()
        model = build_model(encoders)
        model_train(df, model, encoders, args)
    elif args.mode == 'predict':
        encoders = load_encoders()
        model = build_model(encoders)
        model.load_weights('model_weights.hdf5')
        predictions = model_predict(df, model, encoders)
        predictions.to_csv('predictions.csv', index=False)
