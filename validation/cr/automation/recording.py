from contextlib import contextmanager
from .taper import Tape

session_tape = None
active_tape = None

@contextmanager
def record():
    global active_tape, session_tape

    had_active_tape = True
    if not active_tape:
        had_active_tape = False
        if not session_tape:
            session_tape = Tape()
        active_tape = session_tape
    try:
        yield None
    finally:
        if not had_active_tape:
            active_tape = None

@contextmanager
def avoid_recording():
    global active_tape, session_tape

    prev_session_tape = get_session_tape()
    prev_active_tape = get_active_tape()
    session_tape = None
    active_tape = None
    try:
        yield None
    finally:
        session_tape = prev_session_tape
        active_tape = prev_active_tape

def is_recording():
    global active_tape
    if active_tape:
      return True
    return False

def get_active_tape():
    return active_tape

def set_active_tape(tape):
    global active_tape
    active_tape = tape

def get_session_tape():
    return session_tape