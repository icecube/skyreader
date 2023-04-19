"""Package-wide configuration."""


import logging

PLOT_EXTRA_NOT_INSTALLED_ERROR_MSG = (
    "Plotting packages are not installed -- use 'icecube-skyreader[plots]'"
)


LOGGER = logging.getLogger("skyreader")
