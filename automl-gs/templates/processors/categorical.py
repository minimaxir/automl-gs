{{ field }}_enc = df['{{ field }}'].values


{{ field }}_enc = encoders['{{field}}_labeler'].transform({{ field }}_enc)