# Sign4Good

Bridge the gap for the determined and their obstacles


**Project duration**: 1 week  

## Demo on Desktop/Laptop computers
![Working Gif](https://github.com/Zafirmk/Sign4Good/blob/master/working.gif)

## Inspiration
Today, around one million people use Sign Language as their main way to communicate, according to [https://www.csd.org/](Communication Service for the Deaf). We decided to create an application that will help bridge the gap for those who have impaired hearing.

## What it does
Sign4Good allows users to sign a word (full gesture) which is then translated into text. This allows people to communicate without having to fully learn sign language

## How I built it
The hand tracking application was built with opencv. It segments the hand from the frame using masking techniques. The translation of the sign is done using a deep neural network that uses a CNN which recognizes the features of the image. These features are then fed into an RNN which checks the differences between high level frames.

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 256, 256, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 128, 128, 32)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 128, 128, 64)      18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 64, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 32, 256)       295168    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 16, 16, 256)       0         
_________________________________________________________________
reshape_1 (Reshape)          (None, 16, 4096)          0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 16, 64)            1065216   
_________________________________________________________________
lstm_2 (LSTM)                (None, 16, 32)            12416     
_________________________________________________________________
lstm_3 (LSTM)                (None, 32)                8320      
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
=================================================================
Total params: 1,474,566
Trainable params: 1,474,566
Non-trainable params: 0
_________________________________________________________________

```

## Challenges I ran into
Segmenting the fingers from the hand in the hand detection
Handling the large amounts of data

## Accomplishments that I'm proud of
Being able to achieve a model with high accuracy
Segmenting fingers for the hand tracking

## What I learned
Working with Video Detection 

## What's next for Sign4Good
Train using more words
Train using different sign languages 
