{{ field }}_enc = df['{{ field }}'].values


{{ field }}_enc = encoders['{{field}}_encoder'].transform({{ field }}_enc)