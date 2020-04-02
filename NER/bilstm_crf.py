import os
import copy
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import ner_data
import utils

"""
https://blog.csdn.net/baobao3456810/article/details/83388516
"""

class BiLSTMModel(object):
    def __init__(self, model_config):
        self.config = copy.deepcopy(model_config)
        # 构建模型
        self._build_graph()

        if not os.path.exists(self.config["model_dir"]):
            os.makedirs(self.config["model_dir"])
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    def initialize(self, session):
        ckpt = tf.train.get_checkpoint_state(self.config["model_dir"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return self

    def train(self, session, train_data, dev_data, batch_size=16, epoches=30):
        best_valid_f1 = -1.0
        best_valid_step = None
        for epoch in range(epoches):
            for batch_data in train_data.iter_batches(batch_size):
                _, global_step, loss = session.run(
                    [self.train_op, self.global_step, self.loss],
                    feed_dict = self._create_feed_dict(batch_data, self.config["keep_prob"])
                )
                if global_step % 100 == 0:
                    # 每100轮计算一次acc，如果比之前的高就存储模型
                    valid_acc = self.evaluate(
                        session,
                        dev_data,
                        train_data.id2char,
                        train_data.id2tag
                    )
                    if valid_acc["accuracy"]["FB1"] > best_valid_f1:
                        best_valid_f1 = valid_acc["accuracy"]["FB1"]
                        best_valid_step = global_step
                        self.saver.save(
                            session,
                            os.path.join(self.config["model_dir"], "bilstm_ner.ckpt"),
                            global_step=global_step,
                        )

    def evaluate(self, session, data, id2char, id2tag):
        # 先预测再评估
        prediction = self.predict(session, data, id2char, id2tag)
        return utils.conlleval(prediction)

    def predict(self, session, data, id2char, id2tag):
        prediction = list()
        # 每个batch的句子预测为标签
        for batch_data in data.iter_batches(16):
            batch_prediction = session.run(
                self.prediction,
                feed_dict=self._create_feed_dict(batch_data, 1.0)
            )

            for length, char_ids, tag_ids, pred_tag_ids in zip(*batch_data, batch_prediction):
                sentence = list()
                for char, tag, pred_tag in zip(char_ids[:length], tag_ids, pred_tag_ids):
                    # 【句子，标签，预测值】
                    sentence.append((id2char[char], id2tag[tag], id2tag[pred_tag]))
                prediction.append(sentence)
        return prediction

    def _build_graph(self):
        self._build_model()
        self._build_loss()
        self._build_optimizer()

    def _build_model(self):
        with tf.name_scope("input"):
            self.sequence_lengths = tf.placeholder(
                tf.int32,
                shape=(None,),
                name="sequence_lengths"
            )
            self.char_ids = tf.placeholder(
                tf.int32,
                shape=(None, None),
                name="char_ids"
            )
            self.labels = tf.placeholder(
                tf.int32,
                shape=(None, None),
                name="labels"
            )
            self.keep_prob = tf.placeholder(tf.float32, name="ph_keep_prob")

        with tf.variable_scope("embeddings"):
            self.char_embeddings = tf.get_variable(
                "char_embeddings",
                shape=(self.config['chars_num'], self.config['char_dim']),
                initializer=tf.contrib.layers.xavier_initializer()
            )

        self.sequence_embeddings = tf.nn.embedding_lookup(
            self.char_embeddings,
            self.char_ids
        )

        self.lstm_inputs = tf.nn.dropout(self.sequence_embeddings, self.keep_prob)
        with tf.variable_scope("BiLSTM"):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                        self.config["lstm_dim"],
                        use_peepholes=True,
                        initializer=tf.contrib.layers.xavier_initializer(),#初始化权重
                        state_is_tuple=True
                    )
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                self.lstm_inputs,
                dtype=tf.float32,
                sequence_length=self.sequence_lengths
            )
        self.lstm_outputs = tf.concat(outputs, axis=2)

        with tf.variable_scope("projection"):
            with tf.variable_scope("hidden"):
                W = tf.get_variable(
                    "W",
                    shape=[self.config['lstm_dim'] * 2, self.config['lstm_dim']],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                b = tf.get_variable(
                    "b",
                    shape=[self.config['lstm_dim']],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                lstm_outputs = tf.reshape(
                    self.lstm_outputs,
                    shape=[-1, self.config['lstm_dim'] * 2]
                )
                hidden = tf.tanh(tf.nn.xw_plus_b(lstm_outputs, W, b))

            with tf.variable_scope("logits"):
                W = tf.get_variable(
                    "W",
                    shape=[self.config["lstm_dim"], self.config["tags_num"]],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                b = tf.get_variable(
                    "b",
                    shape=[self.config["tags_num"]],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                logits = tf.nn.xw_plus_b(hidden, W, b)

        self.batch_size = tf.shape(self.char_ids)[0]
        self.num_steps = tf.shape(self.char_ids)[-1]
        self.logits = tf.reshape(logits, [-1, self.num_steps, self.config["tags_num"]])
        self.prediction = tf.argmax(self.logits, axis=-1, name="prediction")

    def _build_loss(self):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                    tag_indices=self.labels,
                                                                    sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)

    def _build_optimizer(self):
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.optimizer = tf.train.GradientDescentOptimizer(
                                            self.config["learning_rate"])
            elif optimizer == "adam":
                self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])

            elif optimizer == "adgrad":
                self.optimizer = tf.train.AdagradOptimizer(self.config["learning_rate"])

            else:
                raise KeyError("Unsupported optimizer '%s'" % (optimizer))

            grads_vars = [
                [tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                for g, v in self.optimizer.compute_gradients(self.loss)
            ]
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.train_op = self.optimizer.apply_gradients(grads_vars, self.global_step)

    def _create_feed_dict(self, batch_data, keep_prob):
        sequence_lengths, sequence_char_ids, sequence_tag_ids = batch_data
        feed_dict = {
            self.sequence_lengths: sequence_lengths,
            self.char_ids: sequence_char_ids,
            self.labels: sequence_tag_ids,
            self.keep_prob: keep_prob,
        }
        return feed_dict


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    train_data = ner_data.NERData("data_in/MSRA/train_data")

    # {'<PAD>': 0, '<UNK>': 1, '，': 2, '的': 3, '。': 4, '国': 5, '、': 6, '一': 7, '在': 8, '中': 9, '了': 10, '是': 11...}
    char2id = train_data.char2id
    # {'B-LOC': 0, 'B-ORG': 1, 'B-PER': 2, 'I-LOC': 3, 'I-ORG': 4, 'I-PER': 5, 'O': 6}
    tag2id = train_data.tag2id
    dev_data = ner_data.NERData("data_in/MSRA/dev_data", char2id, tag2id)
    test_data = ner_data.NERData("data_in/MSRA/test_data", char2id, tag2id)

    model_config = {
        "chars_num": len(char2id), # 字典大小
        "char_dim": 100,  # 字符向量维度
        "seg_ids_num": 4,
        "seg_id_dim": 20,
        "lstm_dim": 100,
        "tags_num": len(tag2id),
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "clip": 5.0,
        "keep_prob": 0.8,
        "model_dir": "model/crf/"
    }
    with tf.Session() as session:
        model = BiLSTMModel(model_config)
        model.initialize(session)
        model.train(session, train_data, dev_data)

