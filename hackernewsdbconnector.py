import psycopg2

conn = psycopg2.connect(
    dbname="hd64m1ki",
    user="sy91dhb",
    password="g5t49ao",
    host="178.156.142.230",
    port=5432
)

cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())

cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'hacker_news';
""")
tables = cur.fetchall()
print("Hacker News tables:", tables)


cur.execute("SELECT schema_name FROM information_schema.schemata;")
schemas = cur.fetchall()
print("Schemas:", schemas)


