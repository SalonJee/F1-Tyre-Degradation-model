import fastf1
import os
from fastf1 import get_session
from .config import FASTF1_CACHE_DIR

# Enable FastF1 cache
os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

def load_session(year: int, gp_name: str, session_type: str = "R"):
    """
    Load and cache a FastF1 session.
    """
    session = get_session(year, gp_name, session_type)
    session.load()

    print(f"✅ Loaded: {session.event['EventName']} - {session.name}")
    print("Drivers:", session.laps['Driver'].unique())
    
    return session
