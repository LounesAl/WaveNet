"""test speech to text  """
""" inspire from https://github.com/kingstarcraft/speech-to-text-wavenet2"""


import glob
import json
import os
import time

import glog
import tensorflow.compat.v1 as tf

import dataset
import utils
import wavenet
import datetime

"""
flags = tf.app.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/test.record', 'Path to wave file.')
flags.DEFINE_integer('device', 1, 'The device used to test.')
flags.DEFINE_string('ckpt_dir', 'model/v28', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS

"""
def main(_):
  #utils.load(FLAGS.config_path)
  
  
  data=sample_rate=16000,
  num_channel=20,
  vocabulary=[" ", "a", "b", "c", "d", "e", "f", "g",
               "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
               "r", "s", "t", "u", "v", "w", "x", "y", "z"]
  os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
 # with tf.device(FLAGS.device):
  test_dataset = dataset.create('data/v28/test.record', repeat=False, batch_size=1)
  waves = tf.reshape(tf.sparse.to_dense(test_dataset[0]), shape=[1, -1, utils.Data.num_channel])
  labels = tf.sparse.to_dense(test_dataset[1])
  sequence_length = tf.cast(test_dataset[2], tf.int32)
  vocabulary = tf.constant(utils.Data.vocabulary)
  labels = tf.gather(vocabulary, labels)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary))
  decodes, _ = tf.nn.ctc_beam_search_decoder(
    tf.transpose(logits, perm=[1, 0, 2]), sequence_length, merge_repeated=False)
  outputs = tf.gather(vocabulary,  tf.sparse.to_dense(decodes[0]))
  save = tf.train.Saver()
  
  
   #Tensorboard accuracy on test set 
 
  logdir = "logs/fit/validation" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)


  #write on tensorbord
  writer = tf.compat.v1.summary.FileWriter(logdir)
  writer.add_graph(tf.compat.v1.get_default_graph())

  accuracy = None
  accuracy_summary = tf.compat.v1.Summary()
  accuracy_summary.value.add(tag='F1_Score_test', simple_value=accuracy)

  evalutes = {}
  if os.path.exists('model/v28' + '/evaluate.json'):
   evalutes = json.load(open('model/v28'+ '/evaluate.json', encoding='utf-8'))
  
   
  config_cpu = tf.ConfigProto()
  config_cpu.gpu_options.visible_device_list
  #config = tf.ConfigProto(allow_soft_placement=True)
  #config.gpu_options.allow_growth = True
  with tf.Session(config=config_cpu) as sess:
    status = 0
    while True:
      filepaths = glob.glob('model/v28'+ '/ckpt-22999')
      for filepath in filepaths:
        model_path = os.path.splitext(filepath)[0]
        uid = os.path.split(model_path)[-1]
        saver = tf.train.import_meta_graph('model/v28'+'ckpt-22999.meta')
        #saver.restore(sess, model_path)
        print(saver)
        
        if status != 0:
          continue

        else:
          status = 2
          sess.run(tf.global_variables_initializer())
          sess.run(test_dataset[-1])
          saver.restore(sess, model_path)
          evalutes[uid] = {}
          tps, preds, poses, count = 0, 0, 0, 0
          while True:
            try:
              
              y, y_ = sess.run((labels, outputs))
              y = utils.cvt_np2string(y)
              y_ = utils.cvt_np2string(y_)
              tp, pred, pos = utils.evalutes(y_, y)
              tps += tp
              preds += pred
              poses += pos
              
              #write accuracy in tensorboard 
                      
              f1_score_test= 2 * tps / (preds + poses + 1e-10)
              accuracy = f1_score_test
              accuracy_summary.value[0].simple_value = accuracy
              print(accuracy_summary)
              writer_accuracy_train.add_summary(accuracy_summary, count)
              count += 1
            
            #  if count % 1000 == 0:
            #    glog.info('processed %d: tp=%d, pred=%d, pos=%d.' % (count, tps, preds, poses))
            except:
            #  if count % 1000 != 0:
            #    glog.info('processed %d: tp=%d, pred=%d, pos=%d.' % (count, tps, preds, poses))
              break

          evalutes[uid]['tp'] = tps
          evalutes[uid]['pred'] = preds
          evalutes[uid]['pos'] = poses
          evalutes[uid]['f1'] = 2 * tps / (preds + poses + 1e-20)
          json.dump(evalutes, open('model/v28' + '/evalute.json', mode='w', encoding='utf-8'))
        evalute = evalutes[uid]
        glog.info('Evalute %s: tp=%d, pred=%d, pos=%d, f1=%f.' %
                  (uid, evalute['tp'], evalute['pred'], evalute['pos'], evalute['f1']))
      if status == 1:
        time.sleep(60)
        
      status = 1


if __name__ == '__main__':
  tf.app.run()
"""
if uid in evalutes:
 if status != 0:
            continue
        """
