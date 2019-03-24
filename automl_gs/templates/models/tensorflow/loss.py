{% if problem_type == 'regression' %}
"{{ params['reg_objective'] }}"
{% elif problem_type == 'binary_classification' %}
"binary_crossentropy"
{% else %}
"categorical_crossentropy"
{% endif %}
