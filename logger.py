import logging
import time
'''
levels:
logging.debug()
logging.info()
logging.warning()
logging.error()
logging.critical()
'''

class Logger:
    def __init__(self, logger_name, level='info', file_name='logfile.log'):
        # Create a logger
        self.logger = logging.getLogger(logger_name)

        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        if level not in ['debug', 'info', 'warning', 'error', 'critical']:
            raise ValueError('Invalid level')
        level = levels[['debug', 'info', 'warning', 'error', 'critical'].index(level)]
        self.logger.setLevel(level)

        # Create a file handler
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        fh = logging.FileHandler(timestamp + '_' + file_name)
        fh.setLevel(level)

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add the formatter to the handlers
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger