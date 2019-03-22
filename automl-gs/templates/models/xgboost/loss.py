{% if problem_type == 'regression' %}
'{{ params['reg_objective'] }}',
{% elif problem_type == 'binary_classification' %}
'binary:logistic',
{% else %}
'multi:softprob',
'num_class': df['{{ target_field }}'].nunique(),
{% endif %}
