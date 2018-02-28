import serial
from string import ascii_lowercase

class MKSerial(object):
	def __init__(self, port):
		self._baud_rate = 57600
		self._port = port
		self._ser = serial.Serial()
		self._initialize_serial()
		
	def _initialize_serial(self):
		self._ser.baudrate = self._baud_rate
		self._ser.port = self._port
		self._ser.open()

	def send_command(self, key):
		self._ser.write(bytes(key, 'ascii'))