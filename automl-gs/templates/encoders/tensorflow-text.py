# Fit the Tokenizer.

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['title'].values)

with open('model_vocab.json', 'w', encoding='utf8') as outfile:
    json.dump(tokenizer.word_index, outfile, ensure_ascii=False)
