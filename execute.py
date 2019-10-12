import logging
from examples.example_loading import example_loading


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

example_loading()
