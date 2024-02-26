# Using device
When wearing the device always keep port side right and it is worn around the hips on the front of the user.

# Using software
Carefull using virtual environments as it is not always compatible with tensorflow instead use global version of python 3.10 for greatest compatability with tensorflow

# Training data lables and their meaning
0. Miscilanious
1. Backpack drop
2. Sitting up and down
3. Walking
4. Stairs
5. Standing idle

# legacy notes from initial pull of project files
To run the code
1. Run the following lines to download required packages:
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
2. You need to change "Your_IP_Address" in index.html file;
3. You need to also add "your WiFi SSID", "your Passoword" and "your WiFi IP Address" in main.cpp if you want to programm the micro controler.
