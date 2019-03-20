    # Fit the Tokenizer.
    tokenizer = CountVectorizer(max_features=10000)
    tokenizer.fit(pd.concat([
        {% for field, field_raw, _ in text_fields %}
        df['{{ field_raw }}']{{ ", " if not loop.last }}
        {% endfor %}
    ], axis=0).tolist())

    with open(os.path.join('encoders', 'model_vocab.json'),
              'w', encoding='utf8') as outfile:
        vocab = {k: int(v) for k, v in tokenizer.vocabulary_.items()}
        json.dump(vocab, outfile, ensure_ascii=False)

