import logging
from typing import Coroutine, Callable

from src.memory.stm_manager import StmMemoryManager

logger = logging.getLogger(__name__)
EventHandlerType = Callable[[], Coroutine[None, None, None]]

mm = StmMemoryManager()

async def shutdown() -> None:
    logger.info('Executing shutdown')
    logger.info('Clearing all sessions')
    await mm.clear_all_sessions()
    logger.info('Finished shutdown')


def custom_shutdown_event_handler() -> EventHandlerType:
    return shutdown