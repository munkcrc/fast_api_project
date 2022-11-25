from sqlalchemy import create_engine, MetaData, schema, event, DDL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


db_name = "test_db"
username = "munk"
password = "test"

SQLALCHEMY_DATABASE_URL = f"postgresql://{username}:{password}@localhost:5432/{db_name}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

metadata = MetaData(schema="tenant")
Base = declarative_base(metadata=metadata)


def init_db():
    for mapper in Base.registry.mappers:
        cls = mapper.class_
        if issubclass(cls, Base):
            table_args = getattr(cls, '__table_args__', None)
            if table_args:
                schema = table_args.get('schema')
                if schema:
                    stmt = f"CREATE SCHEMA IF NOT EXISTS {schema}"
                    event.listen(Base.metadata, 'before_create', DDL(stmt))
    Base.metadata.create_all(bind=engine)


def create_schema(schema_name: str):
    if not engine.dialect.has_schema(engine, schema_name):
        engine.execute(schema.CreateSchema(schema_name))


def get_shared_metadata():
    meta = MetaData()
    for table in Base.metadata.tables.values():
        if table.schema != "tenant":
            table.tometadata(meta)
    return meta
