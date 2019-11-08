import serial
import warnings
import numpy as np
from pyfirmata2 import ArduinoMega as Arduino
from pyfirmata2 import util
import sys
import time

from utils.file_io_utils import *


class SerialComm:
	# variables to control the commands to the door control board
	# close_command_on = False  # ! not used for now as we are closing manually
	# close_initiated = None
	open_command_on = False # keeps track of if we are sending an open command
	open_initiated = None # keeps track of the time at which the open command started
	command_duration = 2  # (s). Once a command has been going for longer than this, stop
	
	speaker_command_on = False # same as above but for command to speaker arduino
	speaker_initiated = None 
	tone_duration = 1 # (s). If you change this change Arduino/speaker.ino accordingly  


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
		print("			... connected")

	def camera_triggers(self):
		# ? Old function to generate camera triggers through firmata. Now obsolete because we are using a dedicare arduino for it
		# it doenst really work, lots of gitter between triggers

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
		# Stream bytes through serial connections to arduino, WIP and not really working as desired
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


		# Get pins for door open and door close [as analog outputs]
		self.speaker_commad_pin = self.arduino.get_pin('d:{}:o'.format(self.arduino_config['tone_pin']))
		self.door_open_pin = self.arduino.get_pin('d:{}:o'.format(self.arduino_config['door_open_pin']))

		# start board iteration
		it = util.Iterator(self.arduino)
		it.start()

	def read_arduino_inputs(self):
		return {k: pin.read() for k,pin in self.arduino_inputs.items()}

	def read_arduino_write_to_file(self, camera_timestamp):
		# Read the state of the arduino inputs and append it to the .csv file with the experimental data
		states = self.read_arduino_inputs()
		sensor_states = states.copy() #keep a clean copy

		states["frame_number"] = self.frame_count
		now = time.time() * 1000
		states["elapsed"] = now - self.exp_start_time
		states["camera_timestamp"] = camera_timestamp

		append_csv_file(self.arduino_inputs_file, states, self.arduino_config["arduino_csv_headers"])

		# clean commands to the door board
		self.clean_door_commands()

		return sensor_states

	def clean_door_commands(self):
		if self.open_command_on:
			if time.time() - self.open_initiated > self.command_duration:
				self.open_command_on = False
				self.open_initiated = None
				self.door_open_pin.write(0.0)
				print("Stopped opening door at {}".format(time.time()))

		if self.speaker_command_on:
			if time.time() - self.speaker_initiated > self.tone_duration:
				self.speaker_command_on = False
				self.speaker_initiated = None
				self.speaker_commad_pin.write(0.0)
				print("Stopped audio at {}".format(time.time()))

				# ! open the door when the audio terminated
				self.open_door()

	def play_tone(self,):
		"""
			[Send a command to the speaker arduino to start playing the tone]
		"""
		if not self.speaker_command_on:
			print("Playing audio at {}".format(time.time()))
			self.speaker_command_on = True
			self.speaker_initiated = time.time()
			self.speaker_commad_pin.write(1.0)	

	def open_door(self,):
		"""
			[Send a command to open the arena door, if it's not already on]
		"""
		if not self.open_command_on:
			print("Opening door at {}".format(time.time()))
			self.open_command_on = True
			self.open_initiated = time.time()
			self.door_open_pin.write(1.0)

	def live_sensors_control(self, sensors_states):
		""" [Get's the latest sensor read outs and controls the state of the arena accordingly. E.g. if pressure > th
				open the door.]
		"""
		# Check which sensors are above the threshold
		above_th = [ch for ch,v in sensors_states.items() if v >= self.live_sensors_ths[ch]]

		if len(above_th) == self.n_sensors:
			self.play_tone()

			# open door is now called when "clean_door_commands" terminates the audio stim
			# self.open_door() # this will be ignored if door is already being opening
