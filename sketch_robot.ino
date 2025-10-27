/*
  "manual" - включить ручной режим
  "auto"   - включить автономный режим
  "forward" - ехать вперед
  "back"    - ехать назад
  "left"    - поворот влево
  "right"   - поворот вправо
  "stop"    - остановиться
  "beep"    - подать звуковой сигнал
*/

const int ENA = 5;
const int ENB = 6;
const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 10;

const int piezoPin = 2;
const int trigPin = 3;
const int echoPin = 4;

const int redPin = 11;
const int greenPin = 12;
const int bluePin = 13;

String command = "";
bool manualMode = true;
int speedMotor = 180;

float leftMotorFactor = 0.7;
float rightMotorFactor = 1.0;

void setup() {
  Serial.begin(9600);

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(piezoPin, OUTPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);

  stopMotors();
  setColor(0, 255, 0);
  beep(2);
}

void loop() {
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "manual") {
      manualMode = true;
      stopMotors();
      setColor(0, 255, 0);
      beep(2);
    }
    else if (command == "auto") {
      manualMode = false;
      stopMotors();
      setColor(0, 0, 255);
      beep(2);
    }
    else if (manualMode) {
      handleManualCommand(command);
    }
  }

  if (!manualMode) {
    autonomousMode();
  }
}

void handleManualCommand(String cmd) {
  String action = "";
  int duration = 0;
  int spaceIndex = cmd.indexOf(' ');
  if (spaceIndex != -1) {
    action = cmd.substring(0, spaceIndex);
    duration = cmd.substring(spaceIndex + 1).toInt();
  } else {
    action = cmd;
  }

  if (action == "forward") {
    moveForward();
  } else if (action == "back") {
    moveBackward();
  } else if (action == "left") {
    turnLeft();
  } else if (action == "right") {
    turnRight();
  } else if (action == "stop") {
    stopMotors();
  } else if (action == "beep") {
    beep(1);
  } else {
  }

  if (duration > 0) {
    delay(duration);
    stopMotors();
  }
}

void autonomousMode() {
  int distance = getDistance();

  if (distance > 20) {
    moveForward();
  } else {
    stopMotors();
    delay(300);
    beep(1);
    moveBackward();
    delay(500);
    stopMotors();
    delay(200);

    turnLeft();
    delay(400);
    stopMotors();
    delay(300);
    int leftDist = getDistance();

    turnRight();
    turnRight();
    delay(400);
    stopMotors();
    delay(300);
    int rightDist = getDistance();

    turnLeft();
    delay(400);
    stopMotors();

    if (leftDist > rightDist) {
      turnLeft();
      delay(400);
    } else {
      turnRight();
      delay(400);
    }
  }
}

void moveForward() {
  analogWrite(ENA, speedMotor * leftMotorFactor);
  analogWrite(ENB, speedMotor * rightMotorFactor);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward() {
  analogWrite(ENA, speedMotor * leftMotorFactor);
  analogWrite(ENB, speedMotor * rightMotorFactor);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void turnLeft() {
  analogWrite(ENA, speedMotor * leftMotorFactor);
  analogWrite(ENB, speedMotor * rightMotorFactor);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void turnRight() {
  analogWrite(ENA, speedMotor * leftMotorFactor);
  analogWrite(ENB, speedMotor * rightMotorFactor);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 20000);
  int distance = duration * 0.034 / 2;
  if (distance == 0) distance = 200;
  return distance;
}

void setColor(int red, int green, int blue) {
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
  analogWrite(bluePin, blue);
}

void beep(int times) {
  for (int i = 0; i < times; i++) {
    tone(piezoPin, 1000, 200);
    delay(250);
  }
}
