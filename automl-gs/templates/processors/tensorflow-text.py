# Transform and pad all text fields.

{% for field in fields % }
{{ field }}_enc = encoders['tokenizer'].texts_to_sequences(df['{{ field }}'].values)
{{ field }}_enc= sequence.pad_sequences({{ field }}_enc, maxlen={{ text_max_length }})
{% endfor %}