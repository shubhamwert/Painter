import tensorflow.keras as K
import tensorflow.keras.datasets.mnist as mn
import matplotlib.pyplot as plt
import tensorflow.keras.layers as L
import tensorflow as tf
import tensorflow.keras.regularizers as R
(X_train,Y_train),(X_test,Y_test)=mn.load_data()

X_train.shape
X_train=X_train/255.
X_train=X_train.reshape([-1,28,28,1])


def Model(in_c,o_c):
    r=tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
    input_=L.Input(shape=in_c)   
    x=L.Conv2D(in_c[-1]*8,kernel_size=3,
                    activation='tanh',
                    padding='same',
                    )(input_)
    x=L.BatchNormalization()(x)
    x=L.MaxPool2D()(x)
    x=L.Conv2D(in_c[-1]*16,kernel_size=3,activation='relu',kernel_regularizer=r)(x)
    x=L.AvgPool2D()(x)
    
    x=L.Conv2D(in_c[-1]*32,kernel_size=3,activation='relu',kernel_regularizer=r)(x)
    x=L.MaxPool2D()(x)
    x=L.ReLU()(x)
    x=L.Dropout(0.3)(x)
    x=L.Flatten()(x)
    output=L.Dense(o_c,activation='softmax',kernel_regularizer=r)(x)
    model=K.Model(inputs=[input_],outputs=[output])
    
    return model



m=Model([28,28,1],10)


LR=0.001
m.compile(
            loss=K.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=K.optimizers.Adam(lr=LR),
            metrics=['accuracy']



)
m.fit(X_train,Y_train,batch_size=64,epochs=10)
m.evaluate(X_test,Y_test,batch_size=64)



m.save('./model')