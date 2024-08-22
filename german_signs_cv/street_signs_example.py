import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from my_utils import split_data, create_generators
from my_utils import order_test_set
from deeplearning_models import streetsigns_model 

            
if __name__ == "__main__":
    
    if False: 
        path_to_data = "/Users/bing/Downloads/german_sign_data/Train"
        path_to_save_train = "/Users/bing/Downloads/german_sign_data/training_data/train"
        path_to_save_val = "/Users/bing/Downloads/german_sign_data/training_data/val"
        val_size = 0.15

        split_data(path_to_data,path_to_save_train,path_to_save_val,val_size) 
    
        path_to_images = "/Users/bing/Downloads/german_sign_data/Test"
        path_to_csv = '/Users/bing/Downloads/german_sign_data/Test.csv'
        order_test_set(path_to_images, path_to_csv)
    
    path_to_train = "/Users/bing/Downloads/german_sign_data/training_data/train"
    path_to_val = "/Users/bing/Downloads/german_sign_data/training_data/val"
    path_to_test = "/Users/bing/Downloads/german_sign_data/Test"
    batch_size = 64
    epochs = 15
    
    train_generator, validation_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes
    
    TRAIN = False 
    TEST = True
    
    if TRAIN: 
        path_to_save_model = "./Models"
        
        ckpt_saver = ModelCheckpoint(
            path_to_save_model, 
            monitor = "val_accuracy",
            mode = "max", 
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )
        
        early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 10)
    
    
        model = streetsigns_model(nbr_classes)
        
        model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics="accuracy")
        
        model.fit(train_generator,
                epochs= epochs,
                batch_size = batch_size, 
                validation_data = validation_generator, 
                callbacks = [ckpt_saver, early_stop]
                )
    
    if TEST:
        model = tf.keras.models.load_model("./Models")
        model.summary()
        
        print("Evaluating validation set")
        model.evaluate(validation_generator)
        
        print("Evaluating test set")
        model.evaluate(test_generator)
        
    

    
    
    
    

    
    
    
            