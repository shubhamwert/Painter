import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.datasets.mnist as mn
(X_train,Y_train),(X_test,Y_test)=mn.load_data()
X_test=X_test/255.
X_test.shape
X_test=X_test.reshape([-1,28,28,1])
m=tf.keras.models.load_model('./model')
m.compile(
            loss=K.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=K.optimizers.Adam(lr=0.001),
            metrics=['sparse_categorical_accuracy']



)
print(m.evaluate(X_test,Y_test,batch_size=128))
