tokenizer = Tokenizer(num_words={{ max_words }})

with open('encoders/model_vocab.json', 'r', encoding='utf8', errors='ignore') as infile:
    tokenizer.vocab = json.load(tokenizer.word_index, infile)
encoders['tokenizer'] = tokenizer