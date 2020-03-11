![](https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/source.gif)

<hr></hr>

[![forthebadge made-with-python](https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/python%20badge.svg)](https://www.python.org/)

# Earthquake Damage Classification
An earthquake is the shaking of the surface of the Earth, resulting from the sudden release of energy in the Earth's lithosphere that creates seismic waves. Earthquakes can range in size from those that are so weak that they cannot be felt to those violent enough to toss people around and destroy whole cities. The seismicity, or seismic activity, of an area is the frequency, type and size of earthquakes experienced over a period of time. This classification model helps us to classify the buildings according to their damages before the earthquake based on the past experiences in that area.

# WORKFLOW
1. Create a python environment.
2. Run the requirements.txt file to install all the dependencies.
3. Make sure all the Files are in the same folders.(Unzip the DATASET.rar and place all the files in the same folder)
4. The File "preprocessfinale11.py" is used for preprocessing the dataset.
5. The File "model.py" has the random forest implementation.
6. The pickle file "RF_model.sav" is the saved model of the implementation.

# ACCURACY 70% 
Due to limited time and computational power. I have used less number of "n_estimators" if anyone is using this project and has a high computational power, please do use gridsearch inorder to find the best parameters for high accuracy.
<p align="center">
  <img width="358" height="64" src="https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/Capture.JPG">
</p>

# INSIGHTS

# HEATMAP
<p align="center">
  <img width="1250" height="1307" src="https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/heat.png">
</p>
# FEATURE IMPORTANCE
<p align="center">
  <img width="476" height="252" src="https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/impo.png">
</p>


[![MIT license](https://github.com/Raahul46/Earthquake-damage-classification/blob/master/Images/mit.svg)](https://lbesson.mit-license.org/)
