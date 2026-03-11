import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus


username = 'postgres'
password = quote_plus('sawantomkar@') 
host = '127.0.0.1'
port = '5433'
database ='delivery_analysis'

engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

dataset = pd.read_csv('data/cleaned_data.csv')
table_name = 'rider_data'
dataset.to_sql(table_name,engine,if_exists='replace',index=False)

print(f'Data successfully loaded into table {table_name} in database {database}')
