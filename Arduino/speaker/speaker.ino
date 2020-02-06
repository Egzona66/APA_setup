const int command_pin = 11; // command from python -> sensors arduino
const int speaker_pin = 6; // output to speaker

int tone_frequency = 500; 
int tone_duration_ms = 1; // !!!! if you change this you will have to change comms.py too!

unsigned long tone_start_time = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(speaker_pin, OUTPUT);
  pinMode(command_pin, INPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  int command_state = digitalRead(command_pin);
  if (command_state == 1){
      if (millis() - tone_start_time > tone_duration_ms){
        tone(speaker_pin, tone_frequency, tone_duration_ms);
        tone_start_time = millis();
      }
  }

}
