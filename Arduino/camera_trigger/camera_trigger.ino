// specify the pins of the camera triggers outputs
int pinCam1 = 13; 
int pinCam2 = 10;

void setup() {
  pinMode(pinCam1, OUTPUT);
  pinMode(pinCam2, OUTPUT);

}

void loop() {
  digitalWrite(pinCam1, HIGH);
  digitalWrite(pinCam2, HIGH);
  delay(2);  // <- change this to modify framerate
  
  digitalWrite(pinCam1, LOW);
  digitalWrite(pinCam2, LOW);
  delay(3); // <- change this to modify framerate
}
