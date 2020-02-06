const int command_pin = 11; // command from python -> sensors arduino
const int speaker_pin = 6; // output to speaker

int tone_frequency = 5000; 
int tone_duration_ms = 1; // !!!! if you change this you will have to change comms.py too!

unsigned long tone_start_time = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(speaker_pin, OUTPUT);
  pinMode(command_pin, INPUT);

 tone(speaker_pin, tone_frequency, tone_duration_ms);

}

void loop() {
  // put your main code here, to run repeatedly:
  int command_state = digitalRead(command_pin);

  tone_start_time = millis();


  if (millis() - tone_start_time > tone_duration_ms){
    // tone(speaker_pin, tone_frequency, tone_duration_ms);
    noTone(speaker_pin);
    tone_start_time = millis();
  }

 


}
