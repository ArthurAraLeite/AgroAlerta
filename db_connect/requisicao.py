from db import buscar_dados

df = buscar_dados("leituras_solo")
print(df.shape)
print(df.head())