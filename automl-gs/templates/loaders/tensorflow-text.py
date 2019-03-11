    # Text
    tokenizer = Tokenizer(num_words=10000)

    with open(os.path.join('encoders', 'model_vocab.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        tokenizer.word_index = json.load(infile)
    encoders['tokenizer'] = tokenizer
    