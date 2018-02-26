import asyncio
import websockets
import time
from threading import Thread

class ActionServer(object):
    def __init__(self, uri, actions):
        self._actions = []
        self._uri = uri
        self._actions = actions

    # start the websocket command server
    def run_command_server(self):
        asyncio.get_event_loop().run_until_complete(self._command_server_loop())
    
    # return next action available
    def _get_next_action(self):
        if len(self._actions) > 0:
            return int(self._actions.pop(0))
        else:
            return None

    # loop that continuously grabs the next action
    # from a list a actions being generated from the ML block
    async def _command_server_loop(self):
        async with websockets.connect(self._uri) as websocket:
            while True: 
                next_action = self._get_next_action()
                if (next_action is not None):
                    await websocket.send(next_action.to_bytes(1, byteorder='big'))
