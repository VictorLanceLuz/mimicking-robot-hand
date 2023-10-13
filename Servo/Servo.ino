#include <Servo.h>

// Servo pins
int thumb = 3;
int index = 5;
int middle = 6;
int ring = 10;
int pinky = 11;

// Servo objects
Servo servoThumb;
Servo servoIndex;
Servo servoMiddle;
Servo servoRing;
Servo servoPinky;

void setup() {
  Serial.begin(9600);
  servoThumb.attach(thumb);
  servoIndex.attach(index);
  servoMiddle.attach(middle);
  servoRing.attach(ring);
  servoPinky.attach(pinky);
}

void loop() {
  servoThumb.write(0);
  servoIndex.write(0);
  servoMiddle.write(0);
  servoRing.write(0);
  servoPinky.write(0);

  delay(1000);
  servoThumb.write(180);
  servoIndex.write(180);
  servoMiddle.write(180);
  servoRing.write(180);
  servoPinky.write(80);
  delay(1000);
}
