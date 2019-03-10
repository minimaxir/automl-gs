    # Text
    tokenizer = Tokenizer(num_words=10000)

    with open(os.path.join('encoders', 'model_vocab.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        tokenizer.vocab = json.load(tokenizer.word_index, infile)
    encoders['tokenizer'] = tokenizer
    