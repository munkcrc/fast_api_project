from pydantic import BaseModel


class BankBase(BaseModel):
    id: str
    # role: Enum
    theme_id: str


class BankCreate(BankBase):
    pass


class Bank(BankBase):

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    id: str
    bank: str
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):

    class Config:
        orm_mode = True


class CompositeBase(BaseModel):
    ifrs9_bench_id: str
    bank: str
    bank_central: str


class CompositeCreate(CompositeBase):
    pass


class Composite(CompositeBase):

    class Config:
        orm_mode = True

