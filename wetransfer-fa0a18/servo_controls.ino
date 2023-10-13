#include <Servo.h>

// PWM pins on the nano
const int THUMB_PIN = 3;
const int INDEX_PIN = 5;
const int MIDDLE_PIN = 6;
const int RING_PIN = 10;
const int PINKY_PIN = 11;

// Servo objects for each finger
Servo thumb;
Servo index;
Servo middle;
Servo ring;
Servo pinky;

// Other global variables
String incoming = "";
const String OPEN = "o";
const String CLOSED = "c";

void setup() {
  // Opens the serial port with data rate at 9600 bps
  Serial.begin(9600);
  Serial.setTimeout(100);
  
  // Attaching the servo objects to their respective pins
  thumb.attach(THUMB_PIN);
  index.attach(INDEX_PIN);
  middle.attach(MIDDLE_PIN);
  ring.attach(RING_PIN);
  pinky.attach(PINKY_PIN);

  thumb.write(30);
  index.write(20);
  middle.write(0);
  ring.write(0);
  pinky.write(0);
}

void loop() {
  // Only act when data is received
  if (Serial.available() > 0) {
    incoming = Serial.readString(); // Take the incoming data
    if (incoming == CLOSED){  // If 0 put the hand in the closed position
      //Serial.print('CLOSED');
      thumb.write(180);
      index.write(170);
      middle.write(170);
      ring.write(165);
      pinky.write(160);
    }
    else if(incoming == OPEN){  // If 1 put the hand in the open position
      //Serial.print('CLOSED');
      thumb.write(30);
      index.write(20);
      middle.write(0);
      ring.write(0);
      pinky.write(0);
    }
  }
  delay(250);
}
