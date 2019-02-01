{{ field }}_tf = df['{{ field }}'].values

{% if params['categorical_strat'] != 'all_binary' %}
{{ field }}_counts = df['{{ field }}'].value_counts()

{% if params['categorical_strat'] != 'top10_perc' %}
{{ field }}_perc = floor(0.1 * {{ field }}_counts.size)
{% endif % }

{% if params['categorical_strat'] != 'top50_perc' %}
{{ field }}_perc = floor(0.5 * {{ field }}_counts.size)
{% endif % }

{{ field }}_top = np.array({{ field }}_counts.index[0:{{ field }}_perc], dtype=object)

{{ field }}_labeler = LabelBinarizer()
{{ field }}_labeler.fit({{ field }}_top)

{{ field }}_tf = {{field}} _labeler.transform({{ field }}_tf)
{% endif % }

{% if params['categorical_strat'] == 'all_binary' %}
{{ field }}_labeler = LabelBinarizer()
{{ field }}_tf = {{ field }} _labeler.fit_transform({{ field }}_tf)
{% endif %}

with open('{{ field }}_labeler.json', 'w', encoding='utf8') as outfile:
    json.dump({{ field }}_labeler.classes_ outfile, ensure_ascii=False)