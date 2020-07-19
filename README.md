# Sign4Good (Bay Area Hacks 2020)

<p style="text-align:center;"><img src="https://github.com/Zafirmk/Sign4Good/blob/master/logo.png" alt="Logo"></p>

**Project duration**: 1 Week.   
**IDE**: Google Collab/Visual Studio Code.   
**Python Version**: Python 3.8


## Description
Today, around one million people use Sign Language as their main way to communicate, according to [https://www.csd.org/](CSD) (Communication Service for the Deaf). We decided to create an application that will help bridge the gap for those who have impaired hearing. Sign4Good is an application that allows users to translate their signs into text. In doing so, both parties can easily communicate with each other without having to learn sign language. 

## How it works
1. First a hand tracking software (built with Google Mediapipe) is used to track the movement of the hand.

2. Once a gesture is recorded, the .mp4 file of the gesture is saved as ```testvideo.mp4```  

3. ```testvideo.mp4``` is fed into a trained neural network (CNN connected to an RNN) and the network outputs what the gesture could be

4. Multi-word signing is also supported. The program is written in such a way that it can take the first n frames to be a single gesture and the next n frames to be another gesture. 




Schematic of the CNN-RNN Architechture used in the project

![NetworkImg](https://github.com/Zafirmk/Sign4Good/blob/master/networkimg.png)

## Multi-gesture translation

![MultiGesture Translation](https://github.com/Zafirmk/Sign4Good/blob/master/multigesture.gif)

## What's next for Sign4Good
Train using more words
Train using different sign languages
Implement a word to sign feature

## Getting Started

1. Clone this repo using the following command  
```
$ git clone https://github.com/Zafirmk/Sign4Good.git
$ cd Sign4Good
```
2. Download the [dataset](http://facundoq.github.io/unlp/lsa64/)

3. Change the directories in the source code to your own directories

4. Check [devpost](https://devpost.com/software/sign4good) to see which signs you can try


### Prerequisites
Things you need to install before running:
*  [Python](https://www.python.org/)
*  [OpenCV](https://opencv.org/)
*  [Google Collab](https://colab.research.google.com/)
*  [Keras](https://keras.io/)

#### Additional Notes
*  Datasets obtained from [Facundoq](http://facundoq.github.io/unlp/lsa64//)

