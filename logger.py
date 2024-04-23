import logging

class Logger:
    def __init__(self, logger_name, level=logging.INFO, file_name='logfile.log'):
        # Create a logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # Create a file handler
        fh = logging.FileHandler(file_name)
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