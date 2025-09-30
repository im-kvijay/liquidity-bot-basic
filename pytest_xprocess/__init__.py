from xprocess import ProcessStarter, XProcess, XProcessInfo, XProcessResources


def getrootdir(config):
    return str(getattr(config, "rootpath", ""))


__all__ = [
    "ProcessStarter",
    "XProcess",
    "XProcessInfo",
    "XProcessResources",
    "getrootdir",
]
