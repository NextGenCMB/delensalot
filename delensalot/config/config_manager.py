# config_manager.py
_global_config = None  # Starts uninitialized

def set_config(config_instance):
    global _global_config
    _global_config = config_instance

def get_config():
    if _global_config is None:
        raise RuntimeError("Config has not been initialized.")
    return _global_config