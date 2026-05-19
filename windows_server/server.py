import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.core import SlaveService

print("MT5 bridge server starting on port 18812...")
t = ThreadedServer(SlaveService, hostname="0.0.0.0", port=18812, reuse_addr=True)
print("Server started on port 18812. Keep this window open.")
t.start()
