void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("Goodnight moon!");
}

void loop() {
  Serial.write("aa");

  if (Serial.available()) {
  }
}
