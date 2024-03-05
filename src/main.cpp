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
const char *webSocketServer = "192.168.25.139";

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


    int16_t axRaw, ayRaw, azRaw;
    int16_t gxRaw, gyRaw, gzRaw;
    mpu.getAcceleration(&axRaw, &ayRaw, &azRaw);
    mpu.getRotation(&gxRaw, &gyRaw, &gzRaw);

    float ax, ay, az;
    float gx, gy, gz;
    
    int16_t fgr, far;
    fgr = mpu.getFullScaleGyroRange();
    far = mpu.getFullScaleAccelRange();


    if (fgr == 0) {
      gx = (500.0*gxRaw)/65536.0;
      gy = (500.0*gyRaw)/65536.0;
      gz = (500.0*gzRaw)/65536.0;
    }
    else if (fgr == 1) {
      gx = (1000.0*gxRaw)/65536.0;
      gy = (1000.0*gyRaw)/65536.0;
      gz = (1000.0*gzRaw)/65536.0;
    }
    else if (fgr == 2) {
      gx = (2000.0*gxRaw)/65536.0;
      gy = (2000.0*gyRaw)/65536.0;
      gz = (2000.0*gzRaw)/65536.0;
    }
    else if (fgr == 3) {
      gx = (4000.0*gxRaw)/65536.0;
      gy = (4000.0*gyRaw)/65536.0;
      gz = (4000.0*gzRaw)/65536.0;
    }

    if (far == 0) {
      ax = (4.0*axRaw)/65536.0;
      ay = (4.0*ayRaw)/65536.0;
      az = (4.0*azRaw)/65536.0;
    }
    else if (far == 1) {
      ax = (8.0*axRaw)/65536.0;
      ay = (8.0*ayRaw)/65536.0;
      az = (8.0*azRaw)/65536.0;
    }
    else if (far == 2) {
      ax = (16.0*axRaw)/65536.0;
      ay = (16.0*ayRaw)/65536.0;
      az = (16.0*azRaw)/65536.0;
    }
    else if (far == 3) {
      ax = (32.0*axRaw)/65536.0;
      ay = (32.0*ayRaw)/65536.0;
      az = (32.0*azRaw)/65536.0;
    }

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(ax,4) +
                     ",\"acceleration_y\":" + String(ay,4) +
                     ",\"acceleration_z\":" + String(az,4) +
                     ",\"gyroscope_x\":" + String(gx,4) +
                     ",\"gyroscope_y\":" + String(gy,4) +
                     ",\"gyroscope_z\":" + String(gz,4) + "}";

    Serial.println("Skiter....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(50); // Adjust delay as needed
  }
}
