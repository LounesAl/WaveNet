import os
import glog
import tensorflow as tf
import utils
import wavenet
import dataset
import datetime

flags = tf.compat.v1.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/test.record', 'Filepath to train dataset record.')
flags.DEFINE_integer('batch_size', 8, 'Batch size of train.') #32
flags.DEFINE_integer('display', 100, 'Step to display loss.')
flags.DEFINE_integer('snapshot', 1000, 'Step to save model.')
flags.DEFINE_integer('device', 0, 'The device used to train.')
flags.DEFINE_string('pretrain_dir', 'pretrain', 'Directory to pretrain.')
flags.DEFINE_string('ckpt_path', 'model/v28/ckpt', 'Path to directory holding a checkpoint.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate of train.')
FLAGS = flags.FLAGS


def main(_):
  print('branis1')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
  utils.load('config/english-28.json')
  global_step = tf.compat.v1.train.get_or_create_global_step()
  train_dataset = dataset.create( 'data/v28/test.record', 8, repeat=True)
  print('branis2')
  # bug tensorflow!!!  the  train_dataset[0].shape[0] != FLAGS.batch_size once in a while
  # waves = tf.reshape(tf.sparse.to_dense(train_dataset[0]), shape=[FLAGS.batch_size, -1, utils.Data.num_channel])
  waves = tf.sparse.to_dense(train_dataset[0])
  waves = tf.reshape(waves, [tf.shape(input=waves)[0], -1, utils.Data.num_channel])
  data=sample_rate=16000,
  num_channel=20,
  vocabulary=[" ", "a", "b", "c", "d", "e", "f", "g",
               "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
               "r", "s", "t", "u", "v", "w", "x", "y", "z"]

  labels = tf.cast(train_dataset[1], tf.int32)
  sequence_length = tf.cast(train_dataset[2], tf.int32)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary), is_training=True)
  loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels, logits, sequence_length, time_major=False))
  print('branis3')
  vocabulary = tf.constant(utils.Data.vocabulary)
  decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(a=logits, perm=[1, 0, 2]), sequence_length=sequence_length)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  labels = tf.gather(vocabulary, tf.sparse.to_dense(labels))

  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss, global_step=global_step)
  print('branis4')
  
  #Tensorboard Loss
  logdir = "logs/fit/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
  writer_loss_train = tf.compat.v1.summary.FileWriter(logdir)
  writer_loss_train.add_graph(tf.compat.v1.get_default_graph())
  

  
  summaries =  tf.compat.v1.summary.merge_all()
  run_metadata = tf.compat.v1.RunMetadata()
  loss_summ = tf.compat.v1.summary.scalar('loss', loss)
  
  #Tensorboard accuracy
  logdir = "logs/fit/test" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)


  #train 
  writer_accuracy_train = tf.compat.v1.summary.FileWriter(logdir)
  writer_accuracy_train.add_graph(tf.compat.v1.get_default_graph())
  
  
  summaries =  tf.compat.v1.summary.merge_all()
  run_metadata = tf.compat.v1.RunMetadata()
  loss_summ = tf.compat.v1.summary.scalar('loss_train', loss)

  
  
  accuracy = None
  accuracy_summary = tf.compat.v1.Summary()
  accuracy_summary.value.add(tag='F1-Score_train', simple_value=accuracy)
  
  # run 
  
  test_dataset,test_outputs,test_labels,test_logits=validation_data()
  
  save = tf.compat.v1.train.Saver(max_to_keep=1000)
  config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  print('branis5')
  losses = tf.Variable(0, dtype=tf.float32) # variable that holds accuracy
  #loss= tf.summary.scalar('Loss', losses) # summary to write to TensorBoard
  
  with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(train_dataset[-1])
    if os.path.exists('pretrain') and len(os.listdir('pretrain')) > 0:
      save.restore(sess, tf.train.latest_checkpoint('pretrain'))
    ckpt_dir = os.path.split('model/v28/ckpt')[0]
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    if len(os.listdir(ckpt_dir)) > 0:
     save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
    print('branis6')
    losses_train, tps_train, preds_train, poses_train = 0, 0, 0, 0

    step=0
    while True:
      
      gp, ll_train, uid_train, ot_train, ls_train,loss_summ_train,_= sess.run((global_step,  labels, train_dataset[3], outputs, loss,loss_summ, optimize))
      
      
      
      tp_train, pred_train, pos_train = utils.evalutes(utils.cvt_np2string(ot_train), utils.cvt_np2string(ll_train))
      
     
      tps_train += tp_train
 
      losses_train += ls_train
     
      preds_train += pred_train
      poses_train += pos_train
      
      print("loss_sum",loss_summ_train)
      print("ls",losses_train)
      
      if gp %100 == 0:
        glog.info("Step %d: loss=%f, tp=%d, pos=%d, pred=%d, f1=%f." %
                  (gp, losses_train if gp == 0 else (losses_train / 100), tps_train, preds_train, poses_train,
                   2 * tps_train / (preds_train + poses_train + 1e-10)))
       
        #write loss in tensorboard 
        writer_loss_train.add_summary(loss_summ_train, gp)
        
        #write accuracy in tensorboard 
        f1_score_train= 2 * tps_train / (preds_train + poses_train + 1e-10)
    
        
       
        
        accuracy = f1_score_train
        accuracy_summary.value[0].simple_value = accuracy
        print(accuracy_summary)
        writer_accuracy_train.add_summary(accuracy_summary, gp)
   
       # acc_sum = tf.compat.v1.summary.scalar('accuracy',f1_score)
       # print(acc_sum)
        #writer_accuracy.add_summary(acc_sum, gp)
        
        step=step+1
       
        losses_train, tps_train, preds_train, poses_train = 0, 0, 0, 0
        
      if (gp+1) % 1000 == 0 and gp != 0:
        save.save(sess,'model/v28/ckpt', global_step=global_step)
        
     
        
      

if __name__ == '__main__':
  tf.compat.v1.app.run()
#https://stackoverflow.com/questions/48195037/how-to-decode-a-tensorflow-summary-string