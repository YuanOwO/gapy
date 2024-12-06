import logging
import multiprocessing as mp
import os
from datetime import datetime as dt
from logging import FileHandler, LogRecord
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

_func_prototype = (
    'def {logger_func_name}(self, message, *args, **kwargs):\n'
    '    if self.isEnabledFor({levelname}):\n'
    '        self._log({levelname}, message, args, **kwargs)'
)


def _add_logger_level(
    levelname: str, level: int, *, func_name: Optional[str] = None
) -> None:
    """

    :type levelname: str
        The reference name of the level, e.g. DEBUG, WARNING, etc
    :type level: int
        Numeric logging level
    :type func_name: str
        The name of the logger function to log to a level, e.g. 'info'
        for log.info(...)
    """

    func_name = func_name or levelname.lower()

    setattr(logging, levelname, level)
    logging.addLevelName(level, levelname)

    exec(
        _func_prototype.format(
            logger_func_name=func_name, levelname=levelname
        ),
        logging.__dict__,
        locals(),
    )
    setattr(logging.Logger, func_name, eval(func_name))


fmt = logging.Formatter(
    '[{asctime}] [ {filename:9}:{lineno:03d} ] [ {processName:11} ({process:5d}) ] [{levelname}] {message}',
    datefmt='%H:%M:%S',
    style='{',
)


def get_handler(log_queue: mp.Queue) -> QueueHandler:
    return QueueHandler(log_queue)


def get_logger(name: str, log_queue: mp.Queue) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.EVERYTHING)
    log.addHandler(get_handler(log_queue))
    return log


class LogListener(mp.Process):
    def __init__(self, log_queue: mp.Queue):
        super().__init__(daemon=True)
        self.log_queue = log_queue

    def prepare(self, record: LogRecord):
        """
        Prepare a record for handling.

        This method just returns the passed-in record. You may want to
        override this method if you need to do any custom marshalling or
        manipulation of the record before passing it to the handlers.
        """
        return record

    def handle(self, record: LogRecord):
        """
        Handle a record.

        This just loops through the handlers offering them the record
        to handle.
        """
        record = self.prepare(record)
        handler = self.handler
        process = record.levelno >= handler.level
        if process:
            handler.handle(record)
            handler.flush()

    def run(self):
        os.makedirs('logs', exist_ok=True)
        filename = os.path.join('logs', dt.now().strftime('%Y%m%d_%H%M%S.log'))
        fh = FileHandler(filename=filename, encoding='utf-8')
        fh.setFormatter(fmt)

        self.handler = fh

        q = self.log_queue
        has_task_done = hasattr(q, 'task_done')

        while True:
            record = q.get(True)
            if record is None:
                if has_task_done:
                    q.task_done()
                break
            self.handle(record)
            if has_task_done:
                q.task_done()

        # self.listener.
