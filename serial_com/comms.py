import serial
import warnings
import numpy as np
from pyfirmata2 import ArduinoMega as Arduino
from pyfirmata2 import util, INPUT, OUTPUT
import sys
import time

from fcutils.file_io.io import append_csv_file


class SerialComm:
	# variables to control the commands to the door control board
	open_command_on = False # keeps track of if we are sending an open command
	open_initiated = None # keeps track of the time at which the open command started
	command_duration = 2  # (s). Once a command has been going for longer than this, stop
	
	speaker_command_on = False # same as above but for command to speaker arduino
	speaker_initiated = None 
	tone_duration = 1 # (s). If you change this change Arduino/speaker.ino accordingly  

	door_status = "closed"
	mouse_on_sensors = False  # it's true if the mouse is currently on all 4 sensors
	mouse_stepped_on_sensors = 0 # records at what time the mouse got on the sensors (in ms)

	t0 = time.time()

	def __init__(self):
		pass

	def get_time(self):
		return round(time.time() - self.t0, 2)

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
		
		self.door_status_pin = self.arduino.analog[self.arduino_config['door_status_pin']]
		# self.door_status_pin = self.arduino.get_pin('d:{}:o'.format(self.arduino_config['door_status_pin']))
		# self.door_status_pin.mode = INPUT
		self.door_status_pin.enable_reporting()

		# start board iteration
		it = util.Iterator(self.arduino)
		it.start()

	def read_door_status(self):
		ds = self.door_status_pin.read()
		if ds > .3:
			self.door_status = "closed"
		else:
			self.door_status = "open"
		return ds

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

		# Get if playing tone or opening door
		states['tone_playing'] = self._tone_command
		states['door_opening'] = self._door_command

		append_csv_file(self.arduino_inputs_file, states, self.arduino_config["arduino_csv_headers"])

		# clean commands to the door board
		self.clean_door_commands()

		return sensor_states

	def clean_door_commands(self):
		if self.open_command_on:
			self._door_command = 0
			if time.time() - self.open_initiated > self.command_duration:
				self.open_command_on = False
				self.open_initiated = None
				self.door_open_pin.write(0.0)
				
				print("Stopped opening door at {}s".format(self.get_time()))
		else:
			self._door_command = 0

		if self.speaker_command_on:
			if time.time() - self.speaker_initiated > self.tone_duration:
				self.speaker_command_on = False
				self.speaker_initiated = None
				self.speaker_commad_pin.write(0.0)
				print("Stopped audio at {}s".format(self.get_time()))
				self._tone_command = -1

				# ! open the door when the audio terminated
				# ? now the door starts opening at the same time as when the speaker comes on
				# self.open_door()
			else:
				self._tone_command = 0
		else: 
			self._tone_command = 0

	def play_tone(self,):
		"""
			[Send a command to the speaker arduino to start playing the tone]
		"""
		if not self.speaker_command_on:
			print("Playing audio at {}s".format(self.get_time()))
			self.speaker_command_on = True
			self._tone_command = 1
			self.speaker_initiated = time.time()
			self.speaker_commad_pin.write(1.0)	

	def open_door(self,):
		"""
			[Send a command to open the arena door, if it's not already on]
		"""
		if not self.open_command_on:
			print("Opening door at {}s".format(self.get_time()))
			self.open_command_on = True
			self._door_command = 1
			self.open_initiated = time.time()
			self.door_open_pin.write(1.0)

	def live_sensors_control(self, sensors_states):
		""" [Get's the latest sensor read outs and controls the state of the arena accordingly. E.g. if pressure > th
				open the door.]

			The sensors are checked only if the door is closed which means that the mouse is 
			on the right part of the arena.

			If the mouse applies enough force to all four sensors, a timer is started.
			If the mouse doesn't get off the sensors and enoughtime has elapsed, the door is opened
			and a tone is played.
		"""

		if self.door_status == "closed" and not self.open_command_on:
			# Check which sensors are above the threshold
			above_th = [ch for ch,v in sensors_states.items() if v >= self.live_sensors_ths[ch]]

			if len(above_th) == self.n_sensors:
				# Mouse is now on sensors
				if not self.mouse_on_sensors: 
					# Mouse just stepped onto sensors
					self.mouse_on_sensors = True
					self.mouse_stepped_on_sensors = time.time()
					print("Mouse on sensors at {}s".format(self.get_time()))


				if (time.time()) - self.mouse_stepped_on_sensors > self.time_on_sensors/1000:
					# Mouse has been on the sensors for long enough					
					self.play_tone()
					self.open_door()
			else: 
				# Mouse not on sensors
				if self.mouse_on_sensors:
					print("Mouse off sensors at {}s".format(self.get_time()))
				self.mouse_on_sensors = False
		else:
			# Check when the mouse gets off the sensors but don't do anything else
			above_th = [ch for ch,v in sensors_states.items() if v >= self.live_sensors_ths[ch]]
			if len(above_th) < self.n_sensors:
				if self.mouse_on_sensors:
					print("Mouse off sensors at {}s".format(self.get_time()))
				self.mouse_on_sensors = False
				
