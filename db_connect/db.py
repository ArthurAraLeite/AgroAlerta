# db.py
from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL = "postgresql://postgres.acloppaxgohqupmdosoi:hanumanhitogaM4@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
engine = create_engine(DATABASE_URL)

def buscar_dados(tabela: str, filtro_sql: str = "") -> pd.DataFrame:
    query = f"SELECT * FROM {tabela} {filtro_sql}"
    return pd.read_sql(query, engine)

def salvar_previsao(tabela_destino: str, registro: dict):
    df = pd.DataFrame([registro])
    df.to_sql(tabela_destino, engine, if_exists="append", index=False)