## **SPEECH EMOTION RECOGNITION(SER) & NLTK POLARITY MODELS**

**Dataset** - I trained using Toronto Emotional Speech set (TESS) dataset having  2800 data points (audio files) and 7 distinct emotions (angry, disgust, fear, happy, neutral pleasant surprise, sad)

Source: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

## Introduction
*Real-time contact centre analytics - using sentiment analysis of audio only, transcript only and/or both to improve customer experience*

Sometimes, it could be particularly difficult to track and perform service recovery for customers who had bad experiences on the call, except on the odd instance where the customer or agent volunteered feedback. A ton of customers have encountered a negative experience with a contact center rep when seeking information or trying to resolve an issue with a product or service. The representative might not have provided the right responses, leaving you frustrated.

Now, imagine if a few minutes later, the customer receives another call or message apologizing for the inconvenience and being provided with the information or resolution needed? How would he/she feel?

This kind of near real-time service recovery can make a significant difference in customer experience and satisfaction delivered by this channel.

<img src="https://github.com/Sarah-Data/Real-time-contact-centre-analytics---Sentiment-Analysis/blob/main/JPEGS/Contact%20Centre.jpg" width="550" height="auto">


For this reason, I have come up with a project that leverages ML models that grab and analyze audio for customer sentiments and classifies into 2 broad categories. 

Here is the process diagram of my project 

<img src="https://github.com/Sarah-Data/Real-time-contact-centre-analytics---Sentiment-Analysis/blob/main/JPEGS/System%20Architecture.jpg" width="800" height="auto">

The project takes a customer call audio recording and analyzes both it’s audio features like tone, pitch etc. for speech emotion recognition, and it’s content (by converting the audio to text, to provide conversational context). This analysis leads to 2 broad sentiment categorizations; positive and negative. Negative interactions can be immediately pushed to a specialized team to perform service recovery.

## Model Flow
I utilized a feedforward neural network  model, a type of artificial neural network where the information flows only in one direction, from the input layer to the output layer. Each neuron in one layer is connected to every neuron in the next layer, and there are no cycles or loops in the network.
The SER model results in 99% accuracy with 10 epochs i.e the neural network was trained for 10 iterations on the audio dataset, and after these iterations, it achieved an accuracy of 99% on the training data.

<img src="https://github.com/Sarah-Data/Real-time-contact-centre-analytics---Sentiment-Analysis/blob/main/JPEGS/Epoch.jpg" width="800" height="auto">

What 99% accuracy means for this model is that, there was only one incorrect prediction among the instances – it classified fear as disgust. See Confusion Matrix

<img src="https://github.com/Sarah-Data/Real-time-contact-centre-analytics---Sentiment-Analysis/blob/main/JPEGS/Confusion%20Matrix.jpg" width="550" height="auto">

## Model Testing
### SER Result
<img src="https://github.com/Sarah-Data/Real-time-contact-centre-analytics---Sentiment-Analysis/blob/main/JPEGS/Audio%20sentiment%20output.jpg" width="550" height="auto">
### NLTK Result
