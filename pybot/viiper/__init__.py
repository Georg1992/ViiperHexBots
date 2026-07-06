"""Pure Python VIIPER TCP client (replaces Go input-bridge).

Provides a direct TCP connection to the VIIPER server for creating
virtual keyboard and mouse devices and sending binary input reports.

Protocol reference: https://github.com/Alia5/VIIPER
"""

from pybot.viiper.client import ViiperClient
from pybot.viiper.stream import DeviceStream

__all__ = ["ViiperClient", "DeviceStream"]
