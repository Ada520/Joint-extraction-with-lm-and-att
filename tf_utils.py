import sys
import os.path
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
sys.path.extend(['/mnt/f/PycharmProjects/multihead_joint_entity_relation_extraction-master', '/mnt/f/PycharmProjects/multihead_joint_entity_relation_extraction-master', '/mnt/d/Program Files/Python/Python36/python36.zip', '/mnt/d/Program Files/Python/Python36/DLLs', '/mnt/d/Program Files/Python/Python36/lib', '/mnt/d/Program Files/Python/Python36', '/mnt/d/Program Files/Python/Python36/lib/site-packages', '/mnt/c/Program Files/JetBrains/PyCharm 2018.3.2/helpers/pycharm_matplotlib_backend'])

import utils
import time
import eval
import tensorflow as tf

class model:
    """Set of classes and methods for training the model and computing the ner and head selection loss"""

    def __init__(self,config,emb_mtx,sess):
        """"Initialize data"""
        self.config=config
        self.emb_mtx=emb_mtx
        self.sess=sess

    def getEvaluator(self):
        if self.config.evaluation_method == "strict" and self.config.ner_classes == "BIO":  # the most common metric
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries_type",
                                                 rel_chunk_eval="boundaries_type")
        elif self.config.evaluation_method == "boundaries" and self.config.ner_classes == "BIO":  # s
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries", rel_chunk_eval="boundaries")
        elif self.config.evaluation_method == "relaxed" and self.config.ner_classes == "EC":  # todo
            return eval.relaxedChunkEvaluator(self.config, rel_chunk_eval="boundaries_type")
        else:
            raise ValueError(
                'Valid evaluation methods : "strict" and "boundaries" in "BIO" mode and "relaxed" in "EC" mode .')


    def train(self,train_data, operations, iter):

            loss = 0

            evaluator = self.getEvaluator()
            start_time = time.time()
            for x_train in utils.generator(train_data, operations.m_op, self.config, train=True):
                _, val, predicted_ner, actual_ner, predicted_rel, actual_rel, _, m_train = self.sess.run(
                    [operations.train_step, operations.obj, operations.predicted_op_ner, operations.actual_op_ner, operations.predicted_op_rel, operations.actual_op_rel, operations.score_op_rel,
                     operations.m_op], feed_dict=x_train)  # sess.run(embedding_init, feed_dict={embedding_placeholder: wordvectors})
                
                if self.config.evaluation_method == "relaxed":
                    evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel, m_train['BIO'])
                else:
                    evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel)

                loss += val

            print('\n****iter %d****' % (iter))
            print('-------Train-------')
            print('loss: %f ' % (loss))

            if self.config.evaluation_method == "relaxed":
                evaluator.computeInfoMacro()
            else:
                evaluator.printInfo()

            elapsed_time = time.time() - start_time
            print("Elapsed train time in sec:" + str(elapsed_time))
            print()



    def evaluate(self,eval_data,operations,set):

        print('-------Evaluate on '+set+'-------')

        evaluator = self.getEvaluator()
        for x_dev in utils.generator(eval_data, operations.m_op, self.config, train=False):
            predicted_ner, actual_ner, predicted_rel, actual_rel, _, m_eval = self.sess.run(
                [operations.predicted_op_ner, operations.actual_op_ner, operations.predicted_op_rel, operations.actual_op_rel, operations.score_op_rel, operations.m_op], feed_dict=x_dev)

            if self.config.evaluation_method == "relaxed":
                evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel, m_eval['BIO'])
            else:
                evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel)

        if self.config.evaluation_method == "relaxed":
            if set == 'test':
                evaluator.computeInfoMacro(printScores=True)
            else:
                evaluator.computeInfoMacro(printScores=True)
            if "other" in [x.lower() for x in self.config.dataset_set_ec_tags]: # if other class exists report score without "Other" class, see previous work on the CoNLL04
                return evaluator.getMacroF1scoresNoOtherClass()[2]
            else:
                return evaluator.getMacroF1scores()[2]

        else:
            evaluator.printInfo(printScores=True)
            return  evaluator.getChunkedOverallAvgF1()



    def get_train_op(self,obj):
        import tensorflow as tf

        if self.config.optimizer == 'Adam':

            optim = tf.train.AdamOptimizer(self.config.learning_rate)

        elif self.config.optimizer == 'Adagrad':
            optim = tf.train.AdagradOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'AdadeltaOptimizer':
            optim = tf.train.AdadeltaOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'GradientDescentOptimizer':
            optim = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        if self.config.gradientClipping == True:

            gvs = optim.compute_gradients(obj)

            new_gvs = self.correctGradients(gvs)

            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in new_gvs]
            train_step = optim.apply_gradients(capped_gvs)


        else:
            train_step = optim.minimize(obj)

        return train_step

    def correctGradients(self,gvs):
        import tensorflow as tf

        new_gvs = []
        for grad, var in gvs:
            # print (grad)
            if grad == None:

                grad = tf.zeros_like(var)

            new_gvs.append((grad, var))
        if len(gvs) != len(new_gvs):
            print("gradient Error")
        return new_gvs

    def broadcasting(self, left, right):
        import tensorflow as tf

        left = tf.transpose(left, perm=[1, 0, 2])
        left = tf.expand_dims(left, 3)

        right = tf.transpose(right, perm=[0, 2, 1])
        right = tf.expand_dims(right, 0)

        B = left + right
        B = tf.transpose(B, perm=[1, 0, 3, 2])

        return B

    def getNerScores(self, rnn_out, hidden_size, n_types=1, dropout_keep_in_prob=1):
        import tensorflow as tf

        u_a = tf.get_variable("u_typ", [hidden_size , self.config.hidden_size_n1])  # [128 32]
        v = tf.get_variable("v_typ", [self.config.hidden_size_n1, n_types])  # [32,1] or [32,10]
        b_s = tf.get_variable("b_typ", [self.config.hidden_size_n1])
        b_c = tf.get_variable("b_ctyp", [n_types])

        mul = tf.einsum('aij,jk->aik', rnn_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]

        sum = mul + b_s
        if self.config.activation=="tanh":
            output = tf.nn.tanh(sum)
        elif self.config.activation=="relu":
            output = tf.nn.relu(sum)

        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aik,kp->aip', output, v) + b_c

        return g

    def getHeadSelectionScores(self, rnn_out, hidden_size,dropout_keep_in_prob=1):
        import tensorflow as tf

        u_a = tf.get_variable("u_a", [hidden_size + self.config.label_embeddings_size, self.config.hidden_size_n1])  # [128 32]
        w_a = tf.get_variable("w_a", [hidden_size + self.config.label_embeddings_size, self.config.hidden_size_n1])  # [128 32]
        v = tf.get_variable("v", [self.config.hidden_size_n1, len(self.config.dataset_set_relations)])  # [32,1] or [32,4]
        b_s = tf.get_variable("b_s", [self.config.hidden_size_n1])

        left = tf.einsum('aij,jk->aik', rnn_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]
        right = tf.einsum('aij,jk->aik', rnn_out, w_a)  # [16 348 64] * #[64 32] = [16 348 32]

        outer_sum = self.broadcasting(left, right)  # [16 348 348 32]
        outer_sum_bias = outer_sum + b_s

        if self.config.activation=="tanh":
            output = tf.tanh(outer_sum_bias)
        elif self.config.activation=="relu":
            output = tf.nn.relu(outer_sum_bias)

        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aijk,kp->aijp', output, v) # [16 348 348 1] or [16 348 348 4]
        g = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], tf.shape(g)[2] * len(self.config.dataset_set_relations)])

        return g


    def computeLoss(self,input_rnn, dropout_embedding_keep,dropout_lstm_keep,dropout_lstm_output_keep,
                    seqlen,dropout_fcl_ner_keep,ners_ids, dropout_fcl_rel_keep,is_train,
                    embedding_ids,pos1_ids, pos2_ids,scoring_matrix_gold, reuse = False):

        with tf.variable_scope("loss_computation", reuse=reuse):
            lossNER = 0.0
            if self.config.use_dropout:
                    input_rnn = tf.nn.dropout(input_rnn, keep_prob=dropout_embedding_keep)
                    #input_rnn = tf.Print(input_rnn, [dropout_embedding_keep], 'embedding:  ', summarize=1000)
            if self.config.use_GRU == False:
                for i in range(self.config.num_lstm_layers):
                    if self.config.use_dropout and i>0:
                        input_rnn = tf.nn.dropout(input_rnn, keep_prob=dropout_lstm_keep)
                        #input_rnn = tf.Print(input_rnn, [dropout_lstm_keep], 'lstm:  ', summarize=1000)

                    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)
                    # Backward direction cell
                    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)

                    (lstm_outputs_fw, lstm_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=lstm_fw_cell,
                        cell_bw=lstm_bw_cell,
                        inputs=input_rnn,
                        sequence_length=seqlen,
                        dtype=tf.float32, scope='BiLSTM' + str(i))

                    # input_rnn = tf.concat(lstm_out, 2)
                    # lstm_output = input_rnn

                if self.config.use_dropout:
                    # lstm_output = tf.nn.dropout(lstm_output, keep_prob=dropout_lstm_output_keep)
                    lstm_outputs_fw = tf.nn.dropout(lstm_outputs_fw, keep_prob=dropout_lstm_output_keep)
                    lstm_outputs_bw = tf.nn.dropout(lstm_outputs_bw, keep_prob=dropout_lstm_output_keep)


                if self.config.lmcost_lstm_gamma > 0.0:
                    lossNER += self.config.lmcost_lstm_gamma * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw,
                                                                                    seqlen, embedding_ids, "separate",
                                                                                    "lmcost_lstm_separate")
                if self.config.lmcost_joint_lstm_gamma > 0.0:
                    lossNER += self.config.lmcost_joint_lstm_gamma * self.construct_lmcost(lstm_outputs_fw,
                                                                                          lstm_outputs_bw,
                                                                                          seqlen, embedding_ids,
                                                                                          "joint", "lmcost_lstm_joint")
                output = tf.concat([lstm_outputs_fw, lstm_outputs_bw], 2)

            if self.config.use_GRU == True:
                gru_cell_forward = tf.contrib.rnn.GRUCell(self.config.gru_size)
                gru_cell_backward = tf.contrib.rnn.GRUCell(self.config.gru_size)

                if self.config.gru_keep_prob < 1:
                    gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward,
                                                                     output_keep_prob=self.config.gru_keep_prob)
                    gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward,
                                                                      output_keep_prob=self.config.gru_keep_prob)
                cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * seqlen)
                cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * seqlen)
                self._initial_state_forward = cell_forward.zero_state(self.config.batchsize, tf.float32)
                self._initial_state_backward = cell_backward.zero_state(self.config.batchsize, tf.float32)

                inputs_forward = input_rnn
                inputs_backward = tf.reverse(input_rnn, [1])
                outputs_forward = []
                state_forward = self._initial_state_forward
                # Bi-GRU layer
                with tf.variable_scope('GRU_FORWARD') as scope:
                    for step in range(seqlen):
                        if step > 0:
                            scope.reuse_variables()
                        (cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
                        outputs_forward.append(cell_output_forward)

                outputs_backward = []

                state_backward = self._initial_state_backward
                with tf.variable_scope('GRU_BACKWARD') as scope:
                    for step in range(seqlen):
                        if step > 0:
                            scope.reuse_variables()
                        (cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :],
                                                                               state_backward)
                        outputs_backward.append(cell_output_backward)

                output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [self.config.batchsize, seqlen, self.config.gru_size])
                output_backward = tf.reverse(
                    tf.reshape(tf.concat(axis=1, values=outputs_backward), [self.config.batchsize, seqlen, self.config.gru_size]),
                    [1])
                output = tf.add(output_forward, output_backward)

            # mask = tf.sequence_mask(seqlen, dtype=tf.float32)
            ner_input = output

            is_training = tf.cond(is_train > 0, lambda: True, lambda: False)
            print(is_training)
            if self.config.self_attention == True:
                ner_input = self.multihead_attention(queries=output,
                                                          keys=output,
                                                          num_units=self.config.hidden_size_lstm * 2,
                                                          num_heads=self.config.num_heads,
                                                          dropout_rate=0.1,
                                                          is_training=is_training,
                                                          causality=False,
                                                          scope="multihead_attention_1",
                                                          reuse=False)

            # loss= tf.Print(loss, [tf.shape(loss)], 'shape of loss is:') # same as scoring matrix ie, [1 59 590]
            if self.config.ner_classes == "EC":
                if self.config.use_GRU == False:
                    nerScores = self.getNerScores(ner_input,self.config.hidden_size_lstm * 2, len(self.config.dataset_set_ec_tags),
                                                    dropout_keep_in_prob=dropout_fcl_ner_keep)
                if self.config.use_GRU == True:
                    nerScores = self.getNerScores(ner_input,self.config.gru_size, len(self.config.dataset_set_ec_tags),
                                                    dropout_keep_in_prob=dropout_fcl_ner_keep)
                label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32,
                                               shape=[len(self.config.dataset_set_ec_tags),
                                                      self.config.label_embeddings_size])
            elif self.config.ner_classes == "BIO":
                print('********************************dataset_set_bio_tags')
                print(self.config.dataset_set_bio_tags)
                if self.config.use_GRU == False:
                    nerScores = self.getNerScores(ner_input, self.config.hidden_size_lstm * 2, len(self.config.dataset_set_bio_tags),
                                                    dropout_keep_in_prob=dropout_fcl_ner_keep)
                if self.config.use_GRU == True:
                    nerScores = self.getNerScores(ner_input, self.config.gru_size, len(self.config.dataset_set_bio_tags),
                                                  dropout_keep_in_prob=dropout_fcl_ner_keep)
                label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32,
                                               shape=[len(self.config.dataset_set_bio_tags),
                                                      self.config.label_embeddings_size])

            # nerScores = tf.Print(nerScores, [tf.shape(ners_ids), ners_ids, tf.shape(nerScores)], 'ners_ids:  ', summarize=1000)


            if self.config.ner_loss == "crf":
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(nerScores, ners_ids, seqlen)
                lossTemp = -log_likelihood
                predNers, viterbi_score = tf.contrib.crf.crf_decode(nerScores, transition_params, seqlen)

            elif self.config.ner_loss == "softmax":
                lossTemp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nerScores, labels=ners_ids)
                predNers = tf.cast(tf.arg_max(nerScores, 2), tf.int32)

            key_masks = tf.cast(tf.sign(tf.abs(predNers - 8)), tf.float32)  # (N, T_k)
            key_masks = tf.multiply(key_masks, lossTemp)  # (h*N, T_k)

            if self.config.use_bias == True:
                lossNER += lossTemp + self.config.weight_b * key_masks
            else:
                lossNER += lossTemp

            if self.config.self_attention == True:
                output = self.multihead_attention(queries=output,
                                                          keys=output,
                                                          num_units=self.config.hidden_size_lstm * 2,
                                                          num_heads=self.config.num_heads,
                                                          dropout_rate=0.1,
                                                          is_training=is_training,
                                                          causality=False,
                                                          scope="multihead_attention_2",
                                                          reuse=False)


            if self.config.label_embeddings_size > 0:
                labels = tf.cond(is_train > 0, lambda: ners_ids, lambda: predNers)
                label_embeddings = tf.nn.embedding_lookup(label_matrix, labels)
                rel_input = tf.concat([output, label_embeddings], axis=2)

            else:
                rel_input = output


            if self.config.use_GRU == False:
                rel_scores = self.getHeadSelectionScores(rel_input, self.config.hidden_size_lstm * 2,
                                                         dropout_keep_in_prob=dropout_fcl_rel_keep)
            if self.config.use_GRU == True:
                rel_scores = self.getHeadSelectionScores(rel_input, self.config.gru_size,
                                                         dropout_keep_in_prob=dropout_fcl_rel_keep)

            lossREL = tf.nn.sigmoid_cross_entropy_with_logits(logits=rel_scores, labels=scoring_matrix_gold)
            probas=tf.nn.sigmoid(rel_scores)
            predictedRel = tf.round(probas)

            return lossNER,lossREL,predNers,predictedRel,rel_scores

    def normalize(self, inputs,
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0.0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += tf.layers.dropout(queries, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
            # outputs += tf.layers.dropout(queries, rate=dropout_rate, training=is_training)

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs

    def construct_lmcost(self, input_tensor_fw, input_tensor_bw, sentence_lengths, target_ids, lmcost_type, name):
        with tf.variable_scope(name):
            # lmcost_max_vocab_size = min(tf.shape(target_ids)[0], self.config.lmcost_max_vocab_size)
            lmcost_max_vocab_size = self.config.lmcost_max_vocab_size
            target_ids = tf.where(tf.greater_equal(target_ids, lmcost_max_vocab_size-1), x=(lmcost_max_vocab_size-1)+tf.zeros_like(target_ids), y=target_ids)
            cost = 0.0
            if lmcost_type == "separate":
                lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:]
                lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,:-1]
                lmcost_fw = self._construct_lmcost(input_tensor_fw[:,:-1,:], lmcost_max_vocab_size, lmcost_fw_mask, target_ids[:,1:], name=name+"_fw")
                lmcost_bw = self._construct_lmcost(input_tensor_bw[:,1:,:], lmcost_max_vocab_size, lmcost_bw_mask, target_ids[:,:-1], name=name+"_bw")
                cost += lmcost_fw + lmcost_bw
            elif lmcost_type == "joint":
                joint_input_tensor = tf.concat([input_tensor_fw[:,:-2,:], input_tensor_bw[:,2:,:]], axis=-1)
                lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:-1]
                cost += self._construct_lmcost(joint_input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids[:,1:-1], name=name+"_joint")
            else:
                raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
            return cost


    def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
        with tf.variable_scope(name):
            lmcost_hidden_layer = tf.layers.dense(input_tensor, self.config.lmcost_hidden_layer_size, activation=tf.tanh, kernel_initializer=self.initializer)
            lmcost_output = tf.layers.dense(lmcost_hidden_layer, lmcost_max_vocab_size, activation=None, kernel_initializer=self.initializer)
            lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lmcost_output, labels=target_ids)
            lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
            return tf.reduce_sum(lmcost_loss)

    def run(self):

        import tensorflow as tf

        # shape = (batch size, max length of sentence, max length of word)
        char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        is_train = tf.placeholder(tf.int32)

        # shape = (batch_size, max_length of sentence)
        word_lengths = tf.placeholder(tf.int32, shape=[None, None])

        embedding_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]
        pos1_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]
        pos2_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]

        token_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]

        entity_tags_ids = tf.placeholder(tf.int32, [None, None])

        scoring_matrix_gold = tf.placeholder(tf.float32, [None, None, None])  # [ batch_size  *   max_sequence]


        tokens = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        BIO = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        entity_tags = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]

        # classes = ...
        seqlen = tf.placeholder(tf.int32, [None])  # [ batch_size ]

        doc_ids = tf.placeholder(tf.string, [None])  # [ batch_size ]


        dropout_embedding_keep = tf.placeholder(tf.float32, name="dropout_embedding_keep")
        dropout_lstm_keep = tf.placeholder(tf.float32, name="dropout_lstm_keep")
        dropout_lstm_output_keep = tf.placeholder(tf.float32, name="dropout_lstm_output_keep")
        dropout_fcl_ner_keep = tf.placeholder(tf.float32, name="dropout_fcl_ner_keep")
        dropout_fcl_rel_keep = tf.placeholder(tf.float32, name="dropout_fcl_rel_keep")
        # gru_keep_prob = tf.placeholder(tf.float32, name="gru_keep_prob")

        self.initializer = None
        if self.config.initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        elif self.config.initializer == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config.initializer == "xavier":
            self.initializer = tf.glorot_normal_initializer()
        else:
            raise ValueError("Unknown initializer")

        embedding_matrix = tf.get_variable(name="embeddings", shape=self.emb_mtx.shape,
                                           initializer=tf.constant_initializer(self.emb_mtx), trainable=False)
        pos1_embedding = tf.get_variable('pos1_embedding', [self.config.pos_num, self.config.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.config.pos_num, self.config.pos_size])


        #####char embeddings

        # 1. get character embeddings
        K = tf.get_variable(name="char_embeddings", dtype=tf.float32,
                            shape=[len(self.config.dataset_set_characters), self.config.char_embeddings_size])
        # shape = (batch, sentence, word, dim of char embeddings)
        char_embeddings = tf.nn.embedding_lookup(K, char_ids)

        # 2. put the time dimension on axis=1 for dynamic_rnn
        s = tf.shape(char_embeddings)  # store old shape


        char_embeddings_reshaped = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.char_embeddings_size])
        word_lengths_reshaped = tf.reshape(word_lengths, shape=[-1])


        char_hidden_size = self.config.hidden_size_char

        # 3. bi lstm on chars
        cell_fw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)

        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                              inputs=char_embeddings_reshaped,
                                                                              sequence_length=word_lengths_reshaped,
                                                                              dtype=tf.float32)
        # shape = (batch x sentence, 2 x char_hidden_size)
        output = tf.concat([output_fw, output_bw], axis=-1)

        # shape = (batch, sentence, 2 x char_hidden_size)
        char_rep = tf.reshape(output, shape=[-1, s[1], 2 * char_hidden_size])

        # concat char embeddings

        word_embeddings = tf.nn.embedding_lookup(embedding_matrix, embedding_ids)
        pos1_emb = tf.nn.embedding_lookup(pos1_embedding, pos1_ids)
        pos2_emb = tf.nn.embedding_lookup(pos2_embedding, pos2_ids)

        input_rnn = word_embeddings
        if self.config.use_position == True:
            print(pos1_emb)
            input_rnn = tf.concat([input_rnn, pos1_emb, pos2_emb], axis=-1)
        if self.config.use_chars == True:
            input_rnn = tf.concat([input_rnn, char_rep], axis=-1)


        embeddings_input=input_rnn


        lossNER, lossREL, predicted_entity_tags_ids, predictedRel, rel_scores = self.computeLoss(input_rnn,
                                                                                dropout_embedding_keep,
                                                                                dropout_lstm_keep,
                                                                                dropout_lstm_output_keep, seqlen,
                                                                                dropout_fcl_ner_keep,
                                                                                entity_tags_ids, dropout_fcl_rel_keep,
                                                                                is_train,
                                                                                embedding_ids,
                                                                                pos1_ids, pos2_ids,
                                                                                scoring_matrix_gold,reuse=False)

        obj = tf.reduce_sum(lossNER) + tf.reduce_sum(lossREL)
        actualRel = tf.round(scoring_matrix_gold)
        #perturb the inputs
        if self.config.use_adversarial==True:
            raw_perturb = tf.gradients(obj, embeddings_input)[0]  # [batch, L, dim]
            normalized_per = tf.nn.l2_normalize(raw_perturb, axis=[1, 2])
            perturb = self.config.alpha * tf.sqrt(tf.cast(tf.shape(input_rnn)[2], tf.float32)) * tf.stop_gradient(
                normalized_per)
            perturb_inputs = embeddings_input + perturb

            lossNER_per, lossREL_per, _, _, _ = self.computeLoss(perturb_inputs,
                                                                 dropout_embedding_keep,
                                                                 dropout_lstm_keep,
                                                                 dropout_lstm_output_keep, seqlen,
                                                                 dropout_fcl_ner_keep,
                                                                 entity_tags_ids, dropout_fcl_rel_keep,
                                                                 is_train,
                                                                 embedding_ids,
                                                                 pos1_ids, pos2_ids,
                                                                 scoring_matrix_gold, reuse=True)


            obj+=tf.reduce_sum(lossNER_per)+tf.reduce_sum(lossREL_per)


        m = {}
        m['isTrain'] = is_train
        m['embeddingIds'] = embedding_ids
        m['pos1_emb'] = pos1_ids
        m['pos2_emb'] = pos2_ids
        m['charIds'] = char_ids
        m['tokensLens'] = word_lengths
        m['entity_tags_ids'] = entity_tags_ids
        m['scoringMatrixGold'] = scoring_matrix_gold
        m['seqlen'] = seqlen
        m['doc_ids'] = doc_ids
        m['tokenIds'] = token_ids
        m['dropout_embedding']=dropout_embedding_keep
        m['dropout_lstm']=dropout_lstm_keep
        m['dropout_lstm_output']=dropout_lstm_output_keep
        m['dropout_fcl_ner']=dropout_fcl_ner_keep
        m['dropout_fcl_rel'] = dropout_fcl_rel_keep
        # m['gru_keep_prob'] = gru_keep_prob
        m['tokens'] = tokens
        m['BIO'] = BIO
        m['entity_tags'] = entity_tags

        return obj, m, predicted_entity_tags_ids, entity_tags_ids, predictedRel, actualRel, rel_scores


class operations():
    def __init__(self,train_step,obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel):

        self.train_step=train_step
        self.obj=obj
        self.m_op = m_op
        self.predicted_op_ner = predicted_op_ner
        self.actual_op_ner = actual_op_ner
        self.predicted_op_rel = predicted_op_rel
        self.actual_op_rel = actual_op_rel
        self.score_op_rel = score_op_rel