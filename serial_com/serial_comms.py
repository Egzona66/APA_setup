import sys
import glob
import serial
import time


def available_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
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
    print(result)

def connect_port(com="COM5"):
    ser = serial.Serial(com, timeout=0)
    ser.baudrate  = 115200
    # ser.open()
    print(ser.name, "is open: ", ser.is_open)
    if ser.is_open: return ser
    else: return None

def test_connection():
    ser = connect_port()

    if ser is None: raise valueError("Could not connect")
    
    # Send ranndom number
    while True:
        print("reading")
        # ser.write(str.encode("1"))
        # time.sleep(1)
        line = ser.readline()
        print(line)
        time.sleep(0.01)

if __name__ == '__main__':
    # print(available_ports())

    test_connection()

