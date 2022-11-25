from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, ARRAY, Date, Enum
from sqlalchemy.orm import relationship

from database import Base

ROLE = enumerate(('BASE', 'MEDIUM', 'PREMIUM', 'ADMIN'))


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column("id", Integer, primary_key=True, nullable=False)
    name = Column("name", String(256), nullable=False, index=True, unique=True)
    schema = Column("schema", String(256), nullable=False, unique=True)
    host = Column("host", String(256), nullable=False, unique=True)

    role = Column('role', String(265), nullable=False, index=True)
    theme_id = Column('theme_id', String(265), nullable=False, index=True)

    # users = relationship("User", back_populates="shared.banks")
    # composites = relationship("Composite", back_populates="shared.banks")

    __table_args__ = ({'schema': 'shared'}, )


# class User(Base):
#     __tablename__ = "users"
#     __table_args__ = {'schema': 'shared'}
#
#     id = Column(String, unique=True, primary_key=True)
#     bank = Column(String, ForeignKey("banks.id"))
#     email = Column(String, unique=True)
#     hashed_password = Column(String)
#
#     banks = relationship("Bank", back_populates="shared.users")
#     user_settings = relationship("UserSettings", back_populates="shared.users")
#
#
# class UserSettings(Base):
#     __tablename__ = "user_settings"
#     __table_args__ = {'schema': 'shared'}
#
#     id = Column(Integer, primary_key=True)
#     user_id = Column(String, ForeignKey("users.id"))
#     segments = Column(ARRAY(String, dimensions=10))
#
#     users = relationship("User", back_populates="shared.user_settings")
#
#
# class Composite(Base):
#     __tablename__ = "composites"
#     __table_args__ = {'schema': 'shared'}
#
#     bench_id = Column(Integer, unique=True, primary_key=True)
#     bank = Column(String, ForeignKey("banks.id"))
#     bank_central = Column(String)
#
#     banks = relationship("Bank", back_populates="shared.composites")
#     pd_benchmarks = relationship("PDBenchmark", back_populates="shared.composites")
#
#
# class IFRS9Benchmark(Base):
#     __tablename__ = "ifrs9_benchmarks"
#     __table_args__ = {'schema': 'shared'}
#
#     bank_id = Column(Integer, ForeignKey('composites.bench_id'), unique=True, primary_key=True)
#     benchmark = Column(Float)
#
#     composites = relationship("Composite", back_populates="shared.pd_benchmarks")
#
#
# class IFRS9Output(Base):
#     __tablename__ = "ifrs9_output"
#     __table_args__ = {'schema': 'tenant_db'}
#
#     timestamp = Column(Date, primary_key=True)
#     customer_id = Column(String, primary_key=True)
#     customer_type = Column(String[])
#     model_id = Column(String)
#     pd_12 = Column(Float)
#     target_12 = Column(Boolean)
#     rating = Column(String)
#     rating_num = Column(Integer)


