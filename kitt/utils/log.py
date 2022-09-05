import logging


def initialize_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s:%(levelname)-4s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
