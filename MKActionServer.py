import asyncio
import websockets
import time

class MKActionServer(object):
    def __init__(self, esp32_url, mk_serial, mode):
        self._mode = mode # 1 = serial, 0 = websockets
        self._mk_serial = mk_serial
        self._esp32_url = esp32_url
        self._last_action = None

        # self._action_map = {
        #     0: ['j'],
        #     1: [],
        #     2: ['a','j'],
        #     3: ['d','j']
        # }

        self._serial_action_map = {
            0: ['j'],
            1: ['a','j'],
            2: ['d','j']
        }

        self._ws_action_map = {
            0: [1],
            1: [1, 2],
            2: [1, 4]
        }

    # start the websocket command server
    def run_command_server(self, actions):
        # physical or emulator
        asyncio.get_event_loop().run_until_complete(self._command_server_loop(actions) if self._mode == 0 else self._command_server_serial_loop(actions))
    
        # both of them
        # asyncio.get_event_loop().run_until_complete(self._command_server_loop(actions))
        # asyncio.get_event_loop().run_until_complete(self._command_server_serial_loop(actions))

    def reset_last_action(self):
        self._last_action = None

    # return next action available
    def _get_next_action(self, actions):
        if len(actions) > 0:
            return int(actions.pop(0))
        else:
            return None

    # loop that continuously grabs the next action
    # from a list a actions being generated from the ML block
    async def _command_server_loop(self, actions):
        async with websockets.connect(self._esp32_url) as websocket:
            while True: 
                next_action_index = self._get_next_action(actions)
                if (next_action_index is not None):
                    next_actions = self._ws_action_map[next_action_index]
                    for action in next_actions:
                        await websocket.send(action.to_bytes(1, byteorder='big'))

    # loop that continuously grabs the next action
    # from a list a actions being generated from the ML block
    async def _command_server_serial_loop(self, actions):
        while True:
            next_action_index = self._get_next_action(actions)

            if (next_action_index is not None):
                for action in self._serial_action_map[next_action_index]:
                    self._mk_serial.send_command(action)
                self._last_action = next_action_index
            elif (self._last_action is not None):
                for action in self._serial_action_map[self._last_action]:
                    self._mk_serial.send_command(action)
            else:
                pass

            time.sleep(0.05)