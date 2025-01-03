import json
from pathlib import Path

class ConfigManager:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            if config_path is None:
                raise ValueError("ConfigManager requires a config path on first initialization.")
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):
        """Lädt die JSON-Konfigurationsdatei."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = json.load(f)

    def get(self, key, default=None):
        """Gibt einen Wert aus der Konfiguration zurück."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k)
            if value is None:
                return default
        return value

    def set(self, key, value):
        """Setzt einen Wert in der Konfiguration."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
