    # {{ field_raw }}
    {{ field }}_encoder= LabelBinarizer()

    with open(os.path.join('encoders', '{{ field }}_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        {{ field }}_encoder.classes_ = json.load(infile)
    encoders['{{ field }}_encoder'] = {{ field }}_encoder