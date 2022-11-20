import os
import glog
import tensorflow.compat.v1 as tf
import utils
import wavenet
import dataset
import json
"""
flags = tf.flags
#flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/train.record', 'Filepath to train dataset record.')
flags.DEFINE_integer('batch_size', 32, 'Batch size of train.')
flags.DEFINE_integer('display', 100, 'Step to display loss.')
flags.DEFINE_integer('snapshot', 1000, 'Step to save model.')
flags.DEFINE_integer('device', 0, 'The device used to train.')
flags.DEFINE_string('pretrain_dir', 'pretrain', 'Directory to pretrain.')
flags.DEFINE_string('ckpt_path', 'model/v28/ckpt', 'Path to directory holding a checkpoint.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate of train.')

FLAGS = flags.FLAGS
"""

def main(_):
  
  os.environ["CUDA_VISIBLE_DEVICES"] = str(str(0))

  print("data")
  global_step = tf.train.get_or_create_global_step()
  train_dataset = dataset.create('data/v28/test.record',1, repeat=True)
  #utils.load("config/english-28.json")
  # utils.load("doc/input")
  data=sample_rate=16000,
  num_channel=20,
  vocabulary=[" ", "a", "b", "c", "d", "e", "f", "g",
               "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
               "r", "s", "t", "u", "v", "w", "x", "y", "z"]

  # bug tensorflow!!!  the  train_dataset[0].shape[0] != FLAGS.batch_size once in a while
  # waves = tf.reshape(tf.sparse.to_dense(train_dataset[0]), shape=[FLAGS.batch_size, -1, utils.Data.num_channel])
  waves = tf.sparse.to_dense(train_dataset[0])
  waves = tf.reshape(waves, [tf.shape(waves)[0], -1, utils.Data.num_channel])


  labels = tf.cast(train_dataset[1], tf.int32)
  sequence_length = tf.cast(train_dataset[2], tf.int32)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary), is_training=True)
  loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, sequence_length, time_major=False))

  vocabulary = tf.constant(utils.Data.vocabulary)
  decodes, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  labels = tf.gather(vocabulary, tf.sparse.to_dense(labels))

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss, global_step=global_step)
  
  # save weights  max to keep means how many weight we wille kept in the folder systems 
  save = tf.train.Saver(max_to_keep=1000)
  #config = tf.ConfigProto(allow_soft_placement=True)
  #config.gpu_options.allow_growth = True
  # The config for CPU usage
  config_cpu = tf.ConfigProto()
  config_cpu.gpu_options.visible_device_list
  with tf.Session(config=config_cpu) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_dataset[-1])
    if os.path.exists('pretrain') and len(os.listdir('pretrain')) > 0:
      save.restore(sess, tf.train.latest_checkpoint('pretrain'))
    ckpt_dir = os.path.split('model/v28/ckpt')[0]
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    if len(os.listdir(ckpt_dir)) > 0:
      save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    losses, tps, preds, poses = 0, 0, 0, 0
    while True:
      gp, ll, uid, ot, ls, _ = sess.run((global_step,  labels, train_dataset[3], outputs, loss, optimize))
      tp, pred, pos = utils.evalutes(utils.cvt_np2string(ot), utils.cvt_np2string(ll))
      tps += tp
      losses += ls
      preds += pred
      poses += pos
      if gp % 100 == 0:
        glog.info("Step %d: loss=%f, tp=%d, pos=%d, pred=%d, f1=%f." %
                  (gp, losses if gp == 0 else (losses / 100), tps, preds, poses,
                   2 * tps / (preds + poses + 1e-10)))
        losses, tps, preds, poses = 0, 0, 0, 0
      if (gp+1) % 1000== 0 and gp != 0:
        # Append the step number to the checkpoint name:
        save.save(sess,'model/v28/ckpt', global_step=global_step)


if __name__ == '__main__':
  tf.app.run()
