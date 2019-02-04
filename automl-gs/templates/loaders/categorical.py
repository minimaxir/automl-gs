{{ field }}_labeler = LabelBinarizer()

with open('{{ field }}_labeler.json', 'r', encoding='utf8', errors='ignore') as infile:
    {{ field }}_labeler._attrs = json.load(infile)
encoders['{{ field }}_labeler'] = {{ field }}_labeler