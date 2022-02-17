# CS5100-21-Team5
Final Project for Team 5:  Artik Bharoliya, Katie  Lowen, Marjan Gohari, Monideep Chakraborti, Zhengtian Zhang

## Status 1 Update:
For the purpose of this project, we will be using TensorFlow and OpenCV to execute our hand gestured controlled snake game. We will also be using MediaPipe, which is a customizable machine learning solutions framework developed by google. MediaPipe is an open-source and cross-platform framework, but most importantly it is very lightweight. Additionally, MediaPipe provides a pre-trained ML model for hand detection, which is why we are drawn to this set up. MediaPipe works with RGB images and as we are using our local webcams (which is traditionally an RGB video) the framework fits naturally into the overall project structure. 

MediaPipe uses 21 key points for each hand detection for example wrist, thumb_cmc, thum_mcp, middle_finger_mcp, pinky_dip, etc. Overall,  each finger is given 4 key points to be used in the detection process. The 10 available pre-trained gestures using the TensorFlow pre-trained model are [‘okay’, ‘peace’, ‘thumbs up’, ‘thumbs down’, ‘call me’, ‘stop’, ‘rock’, ‘live long’, ‘fist’, ‘smile’]. We will use “thumbs up” and ”thumbs down” to direct the snake up and down, “stop” to direct the snake right, and “call me” to direct the snake left. These output keywords will then need to be converted into a keyboard equivalent input in the snake game GUI to be used as input. 

We have begun working with the hand recognition model locally on each of our devices. Because MediaPipe is lightweight for the hand-detection, and the TensorFlow framework for gesture recognition is trained to recognize ten different gestures, we can use this model locally.
