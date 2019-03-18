    # Text
    tokenizer = CountVectorizer(max_features=10000)

    with open(os.path.join('encoders', 'model_vocab.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        tokenizer.vocabulary_ = json.load(infile)
    encoders['tokenizer'] = tokenizer
    