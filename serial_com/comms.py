import serial
import warnings
import numpy as np
from pyfirmata2 import ArduinoMega as Arduino
from pyfirmata2 import util
import sys
import time

from utils.file_io_utils import *


class SerialComm:
	def __init__(self):
		pass

	def get_available_ports(self):
		if sys.platform.startswith('win'):
			ports = ['COM%s' % (i + 1) for i in range(256)]
		elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
			ports = glob.glob('/dev/tty[A-Za-z]*')
		elif sys.platform.startswith('darwin'):
			ports = glob.glob('/dev/tty.*')
		else:
			raise EnvironmentError('Unsupported platform')

		result = []
		for port in ports:
			try:
				s = serial.Serial(port)
				s.close()
				result.append(port)
			except (OSError, serial.SerialException):
				pass
			
		print(result)
		self.available_ports = result

	def connect_serial(self):
		ser = serial.Serial(self.com_port, timeout=0)
		ser.baudrate  = self.baudrate

		print(ser.name, "is open: ", ser.is_open)
		if ser.is_open: self.serial = ser
		else: 
			self.serial is None
			warnings.warn("Could not connet to serial on port: {}".format(self.com_port))

	def connect_firmata(self):
		print("Connecting to arduino... ")
		self.arduino = Arduino(self.com_port)
		# self.arduino.samplingOn(1000 / rate)

		print("			... connected")

	def camera_triggers(self):
		# TODO Keep track of how many triggers were generated, the ITI and print the number of triggers vs number of frames in Camera
		self.n_arduino_triggers = 0

		# Given the desired acquisition framerate, compute the sleep intervals
		frame_s = (1-0.23) / self.acquisition_framerate
		sleep = frame_s / 2

		print("Starting camera trigers with sleep time: ", frame_s)
	
		# Send a TTL pulse to the cameras every n millis, keep doing it forever
		camera_pins = [self.arduino.digital[13], self.arduino.digital[10]]
		
		while True:
			#  Prin ITI
			if self.n_arduino_triggers == 0:
				start = time.time()
			elif self.n_arduino_triggers % 100 == 0:
				now = time.time()
				print("100 arduino triggers in {}s - expected: {}s".format(round(now - start, 2), frame_s*100))
				start = now

			# Trigger cameras
			for pin in camera_pins: 
				pin.write(1)

			# time.sleep(sleep)
			# self.arduino.pass_time(sleep)

			for pin in camera_pins: 
				pin.write(0)
			# time.sleep(sleep)
			self.arduino.pass_time(sleep)


			self.n_arduino_triggers += 1

	def read_serial(self, expected=5):
		self.serial.flushInput()
		self.serial.flushOutput()

		ser_bytes = str(self.serial.readline())
		if len(ser_bytes) <= 5: return None # empty string from arduino

		# Remove extra carachters
		cleaned = ser_bytes.split("'")[1].split("\\")[0].split(";")

		# convert to floats, but only return i we have all data
		try:
			numbers = [float(c) for c in cleaned]
			if len(numbers) == expected: return numbers
			else: raise ValueError(numbers)
		except:
			raise ValueError(cleaned)
			pass 
		
	def setup_pins(self):
		# Given an arduino connected to firmata, create 
		# variables to reference the different pins
		self.arduino_inputs = {k:self.arduino.analog[p] for k,p in self.arduino_config["sensors_pins"].items()}
		for pin in self.arduino_inputs.values(): pin.enable_reporting()

		# start board iteration?
		it = util.Iterator(self.arduino)
		it.start()

	def read_arduino_inputs(self):
		return {k: pin.read() for k,pin in self.arduino_inputs.items()}

	def read_arduino_write_to_file(self, camera_timestamp):
		states = self.read_arduino_inputs()
		states["frame_number"] = self.frame_count
		now = time.time() * 1000
		states["elapsed"] = now - self.exp_start_time
		states["camera_timestamp"] = camera_timestamp

		append_csv_file(self.arduino_inputs_file, states, self.arduino_config["arduino_csv_headers"])
