    # Fit the Tokenizer.
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(pd.concat([
        {% for field, field_raw, _ in text_fields %}
        df['{{ field_raw }}']{{ ", " if not loop.last }}
        {% endfor %}
    ], axis=0).tolist())

    with open(os.path.join('encoders', 'model_vocab.json'),
              'w', encoding='utf8') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

