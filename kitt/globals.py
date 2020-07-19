from .settings import Settings

_global_settings = Settings()


def reset_global_settings():
    global _global_settings
    _global_settings = Settings()


def get_global_settings():
    return _global_settings
