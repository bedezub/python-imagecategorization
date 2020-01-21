# Python Image Categorization

Sentdex Deep Learning with Python, Tensorflow and Keras. Given a sample datasets containing images of cat and dog. Obtained from here: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

The projects contains two files, the imageprocessing.py and imagemodelling.py

# imageprocessing.py 

Images are retrieved from the zip file, we set the image to be grayscale because coloured image will increase the feature from 1 to 3 due to RGB colour scheme if I am not mistaken. I will try to do the coloured version in the future. 

Images from the dataset are reshaped and stored in numpy array before we can run it on our CNN or model. I noticed that my setting the value of IMG_SIZE=100, we might get a sharper image but the model will consume more time during the model training. It is better to set it to values ranging from 40-60 because it wouldn't make much of a difference to be honest. 

Unless we are looking for greater amount of details: fur, line, edges and etc

# imagemodelling.py

This is the engine of our model. There are several layers in the model for our training. I managed to get 0.73 validation accuracy for 3 epochs. As I browse through the comment section, some would suggest that Dense(64) does not required any activation function. It will produced greater amount of accuracy but layer without activation is just not make any sense. In addition, there is other person recommend to use Sigmoid as activation. Through 10 epochs, they got around 0.81 validation accuracy which is great!

However, our CNN Dense(64) is running on relu. So yeah, 0.73 is a great number too!
