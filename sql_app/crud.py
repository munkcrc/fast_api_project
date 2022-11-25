from sqlalchemy.orm import Session

import models
import schemas


def get_user(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"    # TODO: make sure to encode!!!
    db_user = models.User(id=user.id, bank=user.bank, email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_bank(db: Session, bank_id: str):
    return db.query(models.Bank).filter(models.Bank.id == bank_id).first()


def get_banks(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Bank).offset(skip).limit(limit).all()


def create_bank(db: Session, bank: schemas.BankCreate):
    db_item = models.Bank(id=bank.id, theme_id=bank.theme_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

