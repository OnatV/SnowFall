from snow_config import SnowConfig

def print_about():
    print("""
    Snowfall: Snow Simulator
by Onat Vuran, Jackson Stanhope, and Livio D'Agostini
for PBS2023 at ETHZurich
""")

def create_default_config() -> SnowConfig:
    cfg = SnowConfig()
    return cfg