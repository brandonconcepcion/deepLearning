# functional approach : function that returns a model 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.keras import Model 

def functional_model():
        
        my_input = Input(shape=(28, 28, 1))
        convolution1 = Conv2D(32, (3, 3), activation='relu')(my_input)
        convolution2 = Conv2D(64, (3, 3), activation='relu')(convolution1)
        pool1 = MaxPool2D()(convolution2)
        normalizer1 = BatchNormalization()(pool1)
        
        convolution3 = Conv2D(128, (3, 3), activation='relu')(normalizer1)
        pool2 = MaxPool2D()(convolution3)
        normalizer2 = BatchNormalization()(pool2)
        
        globalpool = GlobalAvgPool2D()(normalizer2)
        fully_connected1 = Dense(64, activation='relu')(globalpool)
        fully_connected2 = Dense(10, activation='softmax')(fully_connected1)
        
        model = tf.keras.Model(inputs=my_input, outputs=fully_connected2)
        return model
    


# tensorflow.keras.Model : inherit from this class 
class MyCustomModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        
        self.my_input = Input(shape=(28, 28, 1))
        self.convolution1 = Conv2D(32, (3, 3), activation='relu')
        self.convolution2 = Conv2D(64, (3, 3), activation='relu')
        self.pool1 = MaxPool2D()
        self.normalizer1 = BatchNormalization()
        
        self.convolution3 = Conv2D(128, (3, 3), activation='relu')
        self.pool2 = MaxPool2D()
        self.normalizer2 = BatchNormalization()
        
        self.globalpool = GlobalAvgPool2D()
        self.fully_connected1 = Dense(64, activation='relu')
        self.fully_connected2 = Dense(10, activation='softmax')
        
     
    def call(self, my_input):
         x = self.convolution1(my_input)
         x = self.convolution2(x)
         x = self.pool1(x)
         x = self.normalizer1(x)
         x = self.convolution3(x)
         x = self.pool2(x)
         x = self.normalizer2(x)
         x = self.globalpool(x)
         x = self.fully_connected1(x)
         x = self.fully_connected2(x)
         
         return x

def streetsigns_model(nbr_classes): 
    
    my_input = Input(shape = (60,60,3))
    convolution1 = Conv2D(32, (3, 3), activation='relu')(my_input)
    convolution2 = Conv2D(64, (3, 3), activation='relu')(convolution1)
    pool1 = MaxPool2D()(convolution2)
    normalizer1 = BatchNormalization()(pool1)
    
    convolution3 = Conv2D(128, (3, 3), activation='relu')(normalizer1)
    pool2 = MaxPool2D()(convolution3)
    x = BatchNormalization()(pool2)
    
    #x = Flatten()(x)
    globalpool = GlobalAvgPool2D()(x)
    fully_connected1 = Dense(64, activation='relu')(globalpool)
    fully_connected2 = Dense(nbr_classes, activation='softmax')(fully_connected1)
    
    model = tf.keras.Model(inputs=my_input, outputs=fully_connected2)
    return model



if __name__ == "__main__":  
    
    model = streetsigns_model(10)
    model.summary()
    
    