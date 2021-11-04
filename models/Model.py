import tensorflow as tf
from tensorflow.keras import Sequential    
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


def create_model(n_features):
        # define the model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_features,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    
    return model

def train(X_train,y_train,epochs):
    n_features = X_train.shape[1]
    model = create_model(n_features)
    # Compile the model:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.metrics.mean_squared_error
       
    model.compile(optimizer=optimizer, loss=loss)
    # model training
    callbacks = [EarlyStopping(monitor='val_loss', mode='min',
                           patience=10,
                           restore_best_weights=True)]
    #model fit
    model.fit(X_train, y_train,
          shuffle=True,
          callbacks=callbacks,
          epochs=epochs, validation_split=0.2)
    
    return model