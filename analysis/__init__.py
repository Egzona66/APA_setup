from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback

install_traceback(keep_frames=0, hide_locals=True)

logger.configure(handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}])


from tpd import recorder

recorder.start(base_folder=".", name="logs", timestamp=False)
