import os

if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") is None:
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
