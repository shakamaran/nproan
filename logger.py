import logging
import time
import sys
'''
levels:
logging.debug()
logging.info()
logging.warning()
logging.error()
logging.critical()
'''

class Logger:
    def __init__(self, logger_name, level='info', file_name=None):
        # Create a logger
        self.logger = logging.getLogger(logger_name)

        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        if level not in ['debug', 'info', 'warning', 'error', 'critical']:
            raise ValueError('Invalid level')
        level = levels[['debug', 'info', 'warning', 'error', 'critical'].index(level)]
        self.logger.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%H:%M:%S')

        if file_name is not None:
            #change formatter to include date
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # Create a file handler
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            fh = logging.FileHandler(timestamp + '_' + file_name)
            fh.setLevel(level)
            # Add the formatter to the handler
            fh.setFormatter(formatter)
            # Add the handler to the logger
            self.logger.addHandler(fh)
            # print to sys.stdout to avoid red background in notebook
            fh = logging.StreamHandler(sys.stdout)

        # Create a console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        # Add the formatter to the handler
        ch.setFormatter(formatter)
        # Add the handler to the logger
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger