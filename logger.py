import logging

class Logger:
    def __init__(self, logger_name, level=logging.INFO):
        # Create a logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # Create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add the formatter to the console handler
        ch.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger