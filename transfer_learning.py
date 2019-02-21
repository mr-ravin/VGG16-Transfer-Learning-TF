import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import cv2 as cv
import numpy as np
import os
import time
import random

new_height=224
new_width=224
channels=3
time_file="./time/timefile.txt"

class_label_list=['Class1','Class2','Class3','Class4']

def gen_file(list_name=class_label_list):
  fin_lst=[]
  for i in list_name:
    for root, dirs, files in os.walk("data/" + i):
      for file in files:
        if file.endswith(".jpg"):
          tmp=os.path.join(root, file)
          fin_lst.append([i,tmp])
  random.shuffle(fin_lst)
  return fin_lst

def train(epoch=30,batch_size=50,class_name='1',hidd_val=100): 
 num_lst=len(class_label_list)
 gen_data=gen_file(class_label_list)
 fin_cat_lst=class_label_list
 tmp_gen=[]
 for i in gen_data:
   if i[0] in fin_cat_lst:
     tmp_gen.append(i)
 gen_data=tmp_gen

 class_label_lst=[0]*num_lst
 vgg_model=nets.vgg
 images=tf.placeholder(tf.float32,shape=[None,new_height,new_width,channels],name="image_input")
 class_label=tf.placeholder(tf.float32,shape=[None,num_lst])
 prob=tf.placeholder_with_default(1.0, shape=(),name="keep_prob")
 vgg_model_load=vgg_model.vgg_16(images)
 restorer=tf.train.Saver()

 total_time=0.0
 with tf.Session() as sess:
    lr=0.001
    restorer.restore(sess,"./vgg_16.ckpt")
    graph = tf.get_default_graph()
    pl_5 = graph.get_tensor_by_name('vgg_16/conv4/conv4_3/Relu:0')
    print(pl_5)
    pl_8_0=tf.contrib.layers.flatten(pl_5)
    x = slim.fully_connected(pl_8_0, num_lst, scope='fc/fc_1')
    x_class=x
    x_class_soft=tf.nn.relu(x_class)
    x_class_soft = tf.identity(x_class_soft, name='x_class_soft')

    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_class, labels=class_label))

    tf.summary.scalar('total_loss', total_loss)
    opt=tf.train.AdamOptimizer(lr)
    opt_upper = opt.minimize(total_loss)

    tf.summary.scalar('learning_rate',opt._lr)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    time_data=open(time_file,"a")
    time_data.write(str(time.gmtime())+"\n")
    tmp_time1=time.time()
    for ep in range(epoch):
     it=-batch_size
     left=0
     right=batch_size
     chk=0
     while it<len(gen_data) and chk<=1:
      it=it+batch_size
      list_arry=[]
      c_arry=[]
      file_name_lst=[]
      for im in range(it,right):
        class_label_lst=[0]*num_lst
        file_name_lst.append(gen_data[im][1])
        list_arry.append(cv.resize(cv.imread(gen_data[im][1]),(new_width,new_height)))
        class_label_lst[fin_cat_lst.index(gen_data[im][0])]=1
        c_arry.append(class_label_lst)

      if right+batch_size<len(gen_data):
        right=right+batch_size
      else:
        right=len(gen_data)
        chk=chk+1
      
      _,mrg=sess.run([opt_upper,merged],feed_dict={images:list_arry,class_label:c_arry,prob:1.0})
      summary_writer.add_summary(mrg)
      xi=sess.run(x_class,feed_dict={images:list_arry,class_label:c_arry,prob:1.0})
      cst,xii=sess.run([total_loss,x_class_soft],feed_dict={images:list_arry,class_label:c_arry,prob:1.0})

      tmp_time2=time.time()
      total_time=total_time+abs(tmp_time2-tmp_time1)
      for ty in range(len(xi)):
       print("--------------------------")
       print("file_name:",file_name_lst[ty])
       print("prediction: ",xi[ty])
       print("prediction(without_dropout):",xii[ty])
       print("real: ",c_arry[ty])
       print("--------------------------")
       print("epoch: ",str(ep+1)," / ",str(epoch),"Cost : ",cst)
       print("items: ",str(it)," / ",str(len(gen_data)))
       print("\nrunning time (in hrs):  "+str(float(float(total_time)/float(60*60)))+"\n")
     saver.save(sess,"./saved"+"/trained"+str(ep+1)+"/trained.ckpt")             
     summary_writer.close()
     
 time_data.write(str(time.gmtime())+"\n")
 time_data.close()
train(epoch=30,batch_size=10,class_name='1',hidd_val=40)
print("\nShutdown Start.")
os.system("shutdown -t 001")
