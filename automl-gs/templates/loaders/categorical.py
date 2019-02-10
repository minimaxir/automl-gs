{{ field }}_encoder= LabelBinarizer()

with open('encoders/{{ field }}_encoder.json', 'r', encoding='utf8', errors='ignore') as infile:
    {{ field }}_encoder._attrs = json.load(infile)
encoders['{{ field }}_encoder'] = {{ field }}_encoder