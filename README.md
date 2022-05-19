# Sound-Classification
<h3> Task Explaination </h3>
<p> 
  The task is to classify the sound into different categories. I start by extracting features from sound which, preprocess and divided into train dev        and test set.
</p>

<h3> Dataset: (Urban Sound Classification 8k) </h3>
  <p> Dataset can be download from the link below 
  https://urbansounddataset.weebly.com/urbansound8k.html </p>
  
<h3> Model: </h3>
<p> I have tried experimenting with two different models which was mention below:
    <ol> 
      <li> MLP </li>
      <li> Convolutional Network </li>
    </ol>
</p>

<h3> Imp Library: </h3>
<p> I haved used two very important library which are:
<ol>
  <li> Librosa </li>
  <li> Pytorch </li>
</ol>

Librosa is a music library which help us to extract the data from .wav file, librosa is also extensivly used in preprocessing of the features extracted from .wav file.  Librosa provide vast range of different functions which can be used to manipulate the features into different domain. In this task, I used MFCC.
Pytorch as we all know is library for designing our neural network architecture.

<h3>Challenges & Solution: </h3>
<b> Challenges </b>
<ol>
  <li> Less Dataset. </li>
  <li> Overfitting Problem. </li>
</ol>
<b> Solution To avoid overfitting </b>
<ol> 
  <li> L2 Normalization. </li>
  <li> Data Argumentation. </li>
  <li> Dropout. </li>
  <li> Simpler Network. </li>
</ol>

