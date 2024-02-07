#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_BusIO_Register.h>

// Public internet, allow port in firewall
// Replace with your network credentials
const char *ssid = "Testwork";
const char *password = "password";

// Replace with your WebSocket server address
const char *webSocketServer = "192.168.173.139";

const int webSocketPort = 8000;
const char *webSocketPath = "/";
MPU6050 mpu; // Define the sensor
WebSocketsClient client;
// SocketIOClient socketIO;

bool wifiConnected = false;

void setup()
{
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);


  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  client.begin(webSocketServer, webSocketPort, "/ws");
  Serial.println(client.isConnected());
  wifiConnected = true;
}

void loop()
{
  if (wifiConnected)
  {
    client.loop();

    // Get sensor data
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);
    
    int16_t fgr, far;
    fgr = mpu.getFullScaleGyroRange();
    far = mpu.getFullScaleAccelRange();


    if (fgr == 0) {
      gx = (250*gx)/65536;
      gy = (250*gy)/65536;
      gz = (250*gz)/65536;
    }
    else if (fgr == 1) {
      gx = (500*gx)/65536;
      gy = (500*gy)/65536;
      gz = (500*gz)/65536;
    }
    else if (fgr == 2) {
      gx = (1000*gx)/65536;
      gy = (1000*gy)/65536;
      gz = (1000*gz)/65536;
    }
    else if (fgr == 3) {
      gx = (2000*gx)/65536;
      gy = (2000*gy)/65536;
      gz = (2000*gz)/65536;
    }

    if (far == 0) {
      ax = (2*ax)/65536;
      ay = (2*ay)/65536;
      az = (2*az)/65536;
    }
    else if (far == 1) {
      ax = (4*ax)/65536;
      ay = (4*ay)/65536;
      az = (4*az)/65536;
    }
    else if (far == 2) {
      ax = (8*ax)/65536;
      ay = (8*ay)/65536;
      az = (8*az)/65536;
    }
    else if (far == 3) {
      ax = (16*ax)/65536;
      ay = (16*ay)/65536;
      az = (16*az)/65536;
    }

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(ax) +
                     ",\"acceleration_y\":" + String(ay) +
                     ",\"acceleration_z\":" + String(az) +
                     ",\"gyroscope_x\":" + String(gx) +
                     ",\"gyroscope_y\":" + String(gy) +
                     ",\"gyroscope_z\":" + String(gz) + "}";

    Serial.println("Skiikar....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(10); // Adjust delay as needed
  }
}
