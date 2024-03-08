# Fall Detection Device - Group 6
In this github is the code that was used when developing a fall detection device. An image of the device can be seen as Device im.jpg.
The device uses LSTM to detect whether a fall has occured or not and displays it on a website with a plotted live feed of the device data.

# Using device
When wearing the device always keep port side right and it is worn around the hips on the front of the user for consistent results.
Carefull using virtual environments as it is not always compatible with tensorflow instead use global version of python 3.10 for greatest compatability with tensorflow.
Check that correct ip is inputted in the main.cpp and index.html file in src folder. 
Run server.py in src folder and open website in browser under port: 8000.

# Other notable folders and files
Data analysis and testing contains files used for data analysis and testing, it also has the data labeler where you can display data and label easily.
Legacy contains files that are no longer used or of any intresting
Model detection contains the model training files and the model itself
