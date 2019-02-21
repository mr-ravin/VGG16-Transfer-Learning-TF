import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import cv2 as cv
import numpy as np
import os
import time
import random
import glob

total_epoch=3

new_height=224
new_width=224
channels=3
time_file="./time/timefile.txt"
classes=3
category_list_upper=['Class1','Class2','Class3','Class4']

def preprocess(data_dir='./testdata/', dim_width=224, dim_height=224):
    name_list = glob.glob(data_dir + '*/*.jpg')
    for names in name_list:
        tmp_input = cv.imread(str(names))
        pros_name = names.split('/')[-1]
        tmp_grab = cv.resize(tmp_input, (dim_width, dim_height))
        cv.imwrite(names,tmp_grab)

def gen_file(list_name=category_list_upper):
  fin_lst=[]
  #for i in list_name:
  for root, dirs, files in os.walk("testdata/"):
      for file in files:
        if file.endswith(".jpg"):
          tmp=os.path.join(root, file)
          fin_lst.append([tmp])
  random.shuffle(fin_lst)
  return fin_lst


def infer(lst_arry,epoch=30,batch_size=50,class_name='1'):
 with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("./saved/trained"+str(total_epoch)+"/trained.ckpt.meta")
  new_saver.restore(sess, "./saved/trained"+str(total_epoch)+"/trained.ckpt")
  images=tf.get_default_graph().get_tensor_by_name('image_input:0')
  prob=tf.get_default_graph().get_tensor_by_name('keep_prob:0')
  x_class_soft = tf.get_default_graph().get_tensor_by_name('x_class_soft:0')
  for op in tf.get_default_graph().get_operations():
    pass
  xi=sess.run(x_class_soft,feed_dict={images:lst_arry,prob:1})
  return xi

preprocess()
test_upper_list=["Class1","Class2","Class3","Class4"]
list_img=gen_file(test_upper_list)
ret=[]
gt=[]
for im in list_img:
 input_image=cv.imread(im[0])
 input_image=cv.resize(input_image,(new_height,new_width))
 gt.append(input_image)
res=infer(gt,epoch=30,batch_size=3,class_name='1')
tmp_res=[]
cnt=-1
for i in res:
   i=list(i)
   i_copy=i[:]
   i_category_list_upper=category_list_upper[:]
   cnt=cnt+1
   sum_chk=0
   chk_lst=[]
   sum_chk=max(i)
   rm_num=1
   while rm_num<2:
     rm_num=rm_num+1
     i_copy_max=max(i_copy)
     i_idx=i_copy.index(i_copy_max)
     chk_lst.append(i_category_list_upper[i_idx])
     del i_copy[i_idx]
     del i_category_list_upper[i_idx]
     sum_chk=sum_chk+i_copy_max
   tmp_res.append([list_img[cnt],chk_lst])
cntr=[0]*classes
for e in tmp_res:
  ex=category_list_upper.index(e[1][0])
  cntr[ex]=cntr[ex]+1
maxres=category_list_upper[cntr.index(max(cntr))]
print(maxres)
