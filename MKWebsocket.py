#!/usr/bin/env python

import asyncio
import websockets

async def sendCommand():
    async with websockets.connect('ws://192.168.1.1:80/ws') as websocket:
        while True: 
            name = input("What angle? ")
            await websocket.send(name)
            print("> {}".format(name))

            greeting = await websocket.recv()
            print("< {}".format(greeting))

asyncio.get_event_loop().run_until_complete(sendCommand())