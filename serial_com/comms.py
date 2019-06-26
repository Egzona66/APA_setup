import serial
import warnings



class SerialComm:
	def __init__(self):
		pass

	def get_available_ports(self):
		if sys.platform.startswith('win'):
			ports = ['COM%s' % (i + 1) for i in range(256)]
		elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
			# this excludes your current terminal "/dev/tty"
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
		self.available_ports = results

	def connect_serial(self):
		ser = serial.Serial(self.com_port, timeout=0)
		ser.baudrate  = self.baudrate

		print(ser.name, "is open: ", ser.is_open)
		if ser.is_open: self.serial = ser
		else: 
			self.serial is None
			warnings.warn("Could not connet to serial on port: {}".format(self.com_port))
