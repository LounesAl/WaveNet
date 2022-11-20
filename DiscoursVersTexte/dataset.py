import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def create(filepath, batch_size=1, repeat=False, buffsize=1000):
  def _parse(record):
    keys_to_features = {
      'uid': tf.io.FixedLenFeature([], tf.string),
      'audio/data': tf.io.VarLenFeature(tf.float32),
      'audio/shape': tf.io.VarLenFeature(tf.int64),
      'text': tf.io.VarLenFeature(tf.int64)
    }
    features = tf.io.parse_single_example(
      serialized=record,
      features=keys_to_features
    )
    
    audio = features['audio/data'].values
    shape = features['audio/shape'].values
    audio = tf.reshape(audio, shape)
    audio = tf.sparse.from_dense(audio)
    text = features['text']
    return audio, text, shape[0], features['uid']

  dataset = tf.data.TFRecordDataset(filepath).map(_parse).batch(batch_size=batch_size)
  if buffsize > 0:
    dataset = dataset.shuffle(buffer_size=buffsize)
  if repeat:
    dataset = dataset.repeat()
  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  return tuple(list(iterator.get_next()) + [iterator.initializer])
