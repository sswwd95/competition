
# 실습

import numpy as np
import tensorflow as tf
# tf.set_random_seed(66)

##################################################################
print(tf.executing_eagerly()) # false

tf.compat.v1.disable_eager_execution() 
# 이걸 사용해서 끄면 2점대에서도 사용 가능, 즉시 실행 모드를 끄는 것
print(tf.executing_eagerly())
print(tf.__version__)

# tf114
# False
# False
# 1.14.0

# 파이썬 base
# True
# False
# 2.3.1
# tensorflow 1.15?후반가면서 완벽히 경로 바껴서 warning이 아닌 에러뜬다
#   File "c:\Study\tf114(tensorflow1)\tf17_cnn2_eager.py", line 34, in <module>
#     x = tf.placeholder(tf.float32, [None, 28,28,1])
# AttributeError: module 'tensorflow' has no attribute 'placeholder'
# -> tf.compat.v1.placeholder 으로 바꿔준다 -> 1.대 후반대의 코딩을 하는 것!
#########################################################################

# 1점대 코드 있으면 이렇게 변경해서 써! (tf.compat.v1 넣어주기)


# 1. 데이터

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255.
x_test = x_test.reshape(-1,32,32,3).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32, [None, 32,32,3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 2. 모델구성

# Layer1---------------------------------------------------------------------
w1 = tf.compat.v1.get_variable('w1', shape=[3,3,3,64])
L1 = tf.nn.conv2d(x,w1, strides=[1,1,1,1],padding='SAME') 
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer2----------------------------------------------------------------------
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,64,128])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') 
L2 = tf.nn.selu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer3-----------------------------------------------------------------------
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,128,64])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME') 
L3 = tf.nn.selu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer4------------------------------------------------------------------------
w4 = tf.compat.v1.get_variable('w4', shape=[3,3,64,64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME') 
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Flatten----------------------------------------------------------------------
L_flat = tf.reshape(L4, [-1,2*2*64])

# Layer5(Dense)-----------------------------------------------------------------
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64],
                        initializer= tf.keras.initializers.GlorotNormal())
                    #   initializer=tf.compat.v1.initializers.he_normal())
                    # == initializer=tf.compat.v1.initializers.variance_scaling()
b5 = tf.Variable(tf.compat.v1.random_normal([64]), name = 'b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5) 
# L5 = tf.nn.dropout(L5, keep_prob=0.2)

# Layer6(Dense)-----------------------------------------------------------------
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32], 
                       initializer= tf.keras.initializers.GlorotNormal())
                    # initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.Variable(tf.compat.v1.random_normal([32]), name = 'b6')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6) 
# L6 = tf.nn.dropout(L6, keep_prob=0.2)

# Layer7(Dense)-----------------------------------------------------------------
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10],
                       initializer= tf.keras.initializers.GlorotNormal())
                    # initializer=tf.compat.v1.initializers.he_normal())
b7 = tf.Variable(tf.compat.v1.random_normal([10]), name = 'b7')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7) 

# 컴파일, 훈련

learning_rate = 1e-4

loss = tf.reduce_mean(-tf.reduce_mean(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

# 훈련

training_epochs = 20
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss= 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        l, _ = sess.run([loss,optimizer], feed_dict=feed_dict)
        avg_loss += l/total_batch
    print('epoch : ', '%04d' %(epoch+1),
          'loss = {:.9f}'.format(avg_loss))
print("끝")

prediction = tf.equal(tf.math.argmax(hypothesis, 1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# selu
# acc : 0.4948

# relu
# acc : 0.4325

# lr = 1e-4
# acc : 0.6761

# lr = 1e-3
# acc : 0.6784

# lr = 1e-3, training_epochs = 50
# epoch :  0050 loss = nan
# 끝
# acc : 0.1

# lr = 1e-4, training_epochs = 20t
# epoch :  0020 loss = 0.064943620
# 끝
# acc : 0.6806

# node 숫자 수정
# epoch :  0020 loss = 0.068048836
# 끝
# acc : 0.6847

#  initializer= tf.keras.initializers.GlorotNormal())
# 끝
# acc : 0.6886