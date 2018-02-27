import asyncio
import websockets
import time

class MKActionServer(object):
    def __init__(self, uri, mk_serial, actions, mode):
        self._mode = mode # 1 = serial, 0 = websockets
        self._mk_serial = mk_serial
        self._actions = []
        self._uri = uri
        self._actions = actions
        self._last_action = None

        self._action_map = {
            0: ['j'],
            1: [],
            2: ['a','j'],
            3: ['d','j'],
            4: ['a', 'j'],
            5: ['d','j']
        }

    # start the websocket command server
    def run_command_server(self):
        asyncio.get_event_loop().run_until_complete(self._command_server_loop() if self._mode == 0 else self._command_server_serial_loop())
    
    def reset_last_action(self):
        self._last_action = None

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

    # loop that continuously grabs the next action
    # from a list a actions being generated from the ML block
    async def _command_server_serial_loop(self):
        while True:
            next_action_index = self._get_next_action()

            if (next_action_index is not None):
                for action in self._action_map[next_action_index]:
                    self._mk_serial.send_command(action)
                self._last_action = next_action_index
            elif (self._last_action is not None):
                for action in self._action_map[self._last_action]:
                    self._mk_serial.send_command(action)
            else:
                pass
            
            time.sleep(0.1)