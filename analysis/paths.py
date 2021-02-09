from pathlib import Path

def subdirs(folderpath, pattern="*"):
    """
        returns all sub folders in a given folder matching a pattern
    """
    folderpath = Path(folderpath)
    return [f for f in folderpath.glob(pattern) if f.is_dir()]


def files(folderpath, pattern="*"):
    """
        returns all files folders in a given folder matching a pattern
    """
    folderpath = Path(folderpath)
    return [f for f in folderpath.glob(pattern) if f.is_file()]
