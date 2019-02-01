# Fit the Tokenizer.

tokenizer = Tokenizer(num_words={{ max_words }})

with open('model_vocab.json', 'r', encoding='utf8', errors='ignore') as infile:
    tokenizer.vocab = json.load(tokenizer.word_index, infile)

# Transform and pad all text fields.

{% for field in fields % }
{{ field }}_enc = tokenizer.texts_to_sequences(df['{{ field }}'].values)
{{ field }}_enc= sequence.pad_sequences({{ field }}_enc, maxlen={{ text_max_length }})
{% endfor %}