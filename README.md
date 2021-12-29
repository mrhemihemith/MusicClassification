# MusicClassification

This project works on the concept of Signal Processing and Neural Networks.
This is a test project which first takes in samples of Music and tries to study the unique parameters present in it by converting it into a waveform.
Then this waveform is used to capture the important data that is later fed into a neural network where the network learns to classify the data.
Thus at the end , the program must be able to differentiate the music genre and provide an output with a good accuracy.


Math Behind This Project:

A song or music for example is nothing but a wavform with different characteristics.
Once the waveform of a particular sample music is extracted , we look for similar patterns of characteristics in that waveform that are common to the given class of data.
Once we find that data which is MFCC's in this case we store it in a paricular JSON file and then feed in this data to the neural network.
Signal Processing is the anaylysis, manupulation and extraction of region of intrest from the the waveform of songs,pictures etc.
Fourier transform is used to extract the required waveform  
These signals extracted must be processed depending on the purpose , measurmenr and properties required.
Here Direct Fourier Transform , statistics like arithmetical mean , probability , accuracy , graphs come to play.
