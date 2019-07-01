int pinCam1 = 13;
int pinCam2 = 10;

float fr_sensor = .2;
float fl_sensor = .3;
float hr_sensor = .4;
float hl_sensor = .5;

int fr_pin = 6;
int fl_pin = 7;
int hr_pin = 8;
int hl_pin = 9;

void setup() {
  pinMode(pinCam1, OUTPUT);
  pinMode(pinCam2, OUTPUT);

  pinMode(fr_pin, INPUT);
  pinMode(fl_pin, INPUT);
  pinMode(hr_pin, INPUT);
  pinMode(hl_pin, INPUT);

 Serial.begin(115200);


}

void loop() {
  digitalWrite(pinCam1, HIGH);
  digitalWrite(pinCam2, HIGH);
  delay(10);

  fr_sensor = digitalRead(fr_pin);
  fl_sensor = digitalRead(fl_pin);
  hr_sensor = digitalRead(hr_pin);
  hl_sensor = digitalRead(hl_pin);

  Serial.println(String(1.0)+";"+String(fr_sensor)+";"+String(fl_sensor)+";"+String(hr_sensor)+";"+String(hl_sensor));

  
  digitalWrite(pinCam1, LOW);
  digitalWrite(pinCam2, LOW);
  delay(10);

  Serial.println(String(0.0)+";"+String(fr_sensor)+";"+String(fl_sensor)+";"+String(hr_sensor)+";"+String(hl_sensor));
}
