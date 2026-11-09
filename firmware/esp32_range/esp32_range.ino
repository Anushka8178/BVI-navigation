/*
 * esp32_range.ino - HC-SR04 range sensor for the BVI navigation rig.
 *
 * Board (Arduino IDE): "ESP32 Dev Module". Connected to the laptop by USB.
 * Serial: 115200 baud, one line per reading, about 15 lines a second:
 *     D,<distance_cm>,<esp_millis>     e.g. "D,123.4,56789"
 *     D,-1,<esp_millis>                no echo / out of range
 * Each reading is the median of 3 pings (a timeout counts as "no echo",
 * so 2 or more timeouts give -1).
 *
 * Wiring: the HC-SR04 is powered from the ESP32 VIN pin (5 V from USB),
 * because it did not range reliably at 3.3 V. At 5 V its Echo output is a
 * 5 V pulse, and ESP32 GPIOs are NOT 5 V tolerant (3.3 V max), so Echo goes
 * through a 1k / 2k voltage divider:
 *     V_pin = 5 V * R2 / (R1 + R2) = 5 V * 2k / 3k ~= 3.3 V
 * Trig needs no divider: the ESP32 drives it with 3.3 V, which the HC-SR04
 * reads as HIGH.
 *
 *   HC-SR04 pin | ESP32 pin
 *   ------------+--------------------------------------------------
 *   VCC         | VIN (5 V)
 *   GND         | GND
 *   Trig        | GPIO26
 *   Echo        | R1 = 1k -> GPIO27
 *               | R2 = 2k (two 1k in series) from GPIO27 -> GND
 *
 *       Echo ---[ 1k ]---+--- GPIO27
 *                        |
 *                     [ 1k ]
 *                     [ 1k ]   (R2 = 2k)
 *                        |
 *                       GND
 *
 * Mount at chest height facing forward, next to the phone camera.
 * The laptop side is navigation/range_sensor.py (config.yaml: esp32.serial_port).
 */

const int TRIG_PIN = 26;
const int ECHO_PIN = 27;

const unsigned long PERIOD_MS = 66;          // ~15 readings per second
const unsigned long ECHO_TIMEOUT_US = 30000; // ~5 m round trip
const float MIN_CM = 2.0;                    // HC-SR04 spec range
const float MAX_CM = 400.0;
const float NO_ECHO = -1.0;

unsigned long lastReadingMs = 0;

// One ping; returns distance in cm, or NO_ECHO.
float pingCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long echoUs = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (echoUs == 0) {
    return NO_ECHO;
  }
  float cm = echoUs / 58.0;  // speed of sound 343 m/s, there and back
  if (cm < MIN_CM || cm > MAX_CM) {
    return NO_ECHO;
  }
  return cm;
}

// Median of 3 pings. A missing echo sorts as "infinitely far", so the
// median is only valid when at least 2 of the 3 pings got an echo.
float medianOf3Cm() {
  float v[3];
  for (int i = 0; i < 3; i++) {
    float cm = pingCm();
    v[i] = (cm < 0) ? 1e9 : cm;
    delay(5);  // let the previous echo die out
  }
  // Sort 3 values.
  if (v[0] > v[1]) { float t = v[0]; v[0] = v[1]; v[1] = t; }
  if (v[1] > v[2]) { float t = v[1]; v[1] = v[2]; v[2] = t; }
  if (v[0] > v[1]) { float t = v[0]; v[0] = v[1]; v[1] = t; }
  return (v[1] >= 1e9) ? NO_ECHO : v[1];
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  delay(50);
  Serial.println("# esp32_range ready");  // ignored by the laptop parser
}

void loop() {
  unsigned long now = millis();
  if (now - lastReadingMs < PERIOD_MS) {
    return;
  }
  lastReadingMs = now;

  float cm = medianOf3Cm();
  if (cm < 0) {
    Serial.printf("D,-1,%lu\n", millis());
  } else {
    Serial.printf("D,%.1f,%lu\n", cm, millis());
  }
}
