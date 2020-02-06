/* Minimum_Source*/
#define DXL_BUS_SERIAL1 1  //Dynamixel on Serial1(USART1)  <-OpenCM9.04

#define DXL_BUS_SERIAL2 2  //Dynamixel on Serial2(USART2)  <-LN101,BT210

#define DXL_BUS_SERIAL3 3  //Dynamixel on Serial3(USART3)  <-OpenCM 485EXP



Dynamixel Dxl(DXL_BUS_SERIAL1);

int servo_id = 1;
int speed = 600;
int duration = 1100;
byte temp = 0;  // store the messages sent through USB serials in bytes
int door_status = 0; // keeps track of whether the door is up or down --  0 -> door down // 1 -> door up



void setup() {
  // put your setup code here, to run once:
  Dxl.begin(3);  
  Dxl.wheelMode(servo_id);
}

void loop() {
  temp = (char)SerialUSB.read();
  // SerialUSB.print((char)temp); // uncomment this line if you want to print to serial monitor which letter you sent
    
  // put your main code here, to run repeatedly: 
  // Execute the correct command
  if ((char)temp == 'c' && door_status == 0){  // CLOSING: DOOR GOING DOWN
    SerialUSB.println("Moving the door DOWN");
    Dxl.goalSpeed(servo_id, speed); //forward
    delay(duration); // waiting for door to open
    Dxl.goalSpeed(servo_id, 0); // stopping the servo
    door_status = 1; 
    SerialUSB.println("Ready!!");
  } else if ((char)temp == 'o' && door_status == 1){  // OPENING: DOOR GOING UP
    SerialUSB.println("Moving the door UP");
    Dxl.goalSpeed(servo_id, speed | 0x400); //forward
    delay(duration); // waiting for door to open
    Dxl.goalSpeed(servo_id, 0); // stopping the servo
    door_status = 0;
    SerialUSB.println("Ready!!");
    
  }  else if ((char)temp == 'c' && door_status == 1) {
     SerialUSB.println("Cant move the door DOWN because it's down already!!");
  }  else if ((char)temp == 'o' && door_status == 0) {
     SerialUSB.println("Cant move the door UP because it's up already!!");
  }   
    
    
}

