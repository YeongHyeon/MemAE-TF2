import tensorflow as tf

class MemAE(object):

    def __init__(self, height, width, channel, leaning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.leaning_rate = leaning_rate
        self.ckpt_dir = ckpt_dir

        self.name_bank, self.params_trainable, self.conv_shapes = [], [], []
        self.initializer = tf.initializers.glorot_normal()

        z = self.encoder(x=tf.zeros([1, self.height, self.width, self.channel]), training=False, verbose=True)
        z_hat, _ = self.memory(z, verbose=True)
        x_hat = self.decoder(z_hat, training=False, verbose=True)

        self.optimizer = tf.optimizers.Adam(self.leaning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, iteration=0, train=False):

        with tf.GradientTape() as tape:
            z = self.encoder(x=x, training=train, verbose=False)
            z_hat, w_hat = self.memory(z, verbose=False)
            x_hat = self.decoder(z_hat, training=train, verbose=False)

            mse = tf.reduce_sum(tf.square(x - x_hat), axis=(1, 2, 3))
            mem_etrp = tf.reduce_sum((-w_hat) * tf.math.log(w_hat + 1e-12), axis=(1, 2, 3))
            loss = tf.reduce_mean(mse + (0.0002 * mem_etrp))

        if(train):
            gradients = tape.gradient(loss, self.params_trainable)
            self.optimizer.apply_gradients(zip(gradients, self.params_trainable))

            with self.summary_writer.as_default():
                tf.summary.scalar('MemAE/MSE', tf.reduce_mean(mse), step=iteration)
                tf.summary.scalar('MemAE/Cross_Entropy', tf.reduce_mean(mem_etrp), step=iteration)
                tf.summary.scalar('MemAE/Total_Loss', loss, step=iteration)

        return x_hat, mse, mem_etrp, loss

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.name_bank):
            vars_to_save[self.name_bank[idx]] = self.params_trainable[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.name_bank):
            vars_to_load[self.name_bank[idx]] = self.params_trainable[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def get_weight(self, vshape, transpose=False, bias=True, name=""):

        try:
            idx_w = self.name_bank.index("%s_w" %(name))
            if(bias): idx_b = self.name_bank.index("%s_b" %(name))
        except:
            w = tf.Variable(self.initializer(vshape), \
                name="%s_w" %(name), trainable=True, dtype=tf.float32)
            self.name_bank.append("%s_w" %(name))
            self.params_trainable.append(w)

            if(bias):
                if(transpose): b = tf.Variable(self.initializer([vshape[-2]]), \
                    name="%s_b" %(name), trainable=True, dtype=tf.float32)
                else: b = tf.Variable(self.initializer([vshape[-1]]), \
                    name="%s_b" %(name), trainable=True, dtype=tf.float32)
                self.name_bank.append("%s_b" %(name))
                self.params_trainable.append(b)
        else:
            w = self.params_trainable[idx_w]
            if(bias): b = self.params_trainable[idx_b]

        if(bias): return w, b
        else: return w

    def conv2d(self, inputs, variables, stride_size, padding):

        [weights, biasis] = variables
        out = tf.nn.conv2d(inputs, weights, \
            strides=[1, stride_size, stride_size, 1], padding=padding) + biasis

        return out

    def conv2d_tr(self, inputs, variables, output_shape, stride_size, padding):

        [weights, biasis] = variables
        out = tf.nn.conv2d_transpose(inputs, weights, output_shape, \
            strides=[1, stride_size, stride_size, 1], padding=padding) + biasis

        return out
    
    def batch_norm(self, inputs, name=""):
        
        # https://arxiv.org/pdf/1502.03167.pdf
        
        mean = tf.reduce_mean(inputs)
        std = tf.math.reduce_std(inputs)
        var = std**2
        
        try:
            idx_offset = self.name_bank.index("%s_offset" %(name))
            idx_scale = self.name_bank.index("%s_scale" %(name))
        except:
            offset = tf.Variable(0, \
                name="%s_offset" %(name), trainable=True, dtype=tf.float32)
            self.name_bank.append("%s_offset" %(name))
            self.params_trainable.append(offset)
            scale = tf.Variable(1, \
                name="%s_scale" %(name), trainable=True, dtype=tf.float32)
            self.name_bank.append("%s_scale" %(name))
            self.params_trainable.append(scale)
        else:
            offset = self.params_trainable[idx_offset]
            scale = self.params_trainable[idx_scale]
            
        offset # zero
        scale # one
        out = tf.nn.batch_normalization(
            x = inputs,
            mean=mean,
            variance=var,
            offset=offset,
            scale=scale,
            variance_epsilon=1e-12,
            name=name
        )
        
        return out
        
    def cosine_sim(self, x1, x2):

        num = tf.linalg.matmul(x1, tf.transpose(x2, perm=[0, 1, 3, 2]), name='attention_num')
        denom =  tf.linalg.matmul(x1**2, tf.transpose(x2, perm=[0, 1, 3, 2])**2, name='attention_denum')
        w = (num + 1e-12) / (denom + 1e-12)

        return w

    def encoder(self, x, training=False, verbose=False):

        if(verbose): print("Encoder:", x.shape)
        self.conv_shapes.append(x.shape)
        encode1 = self.conv2d(inputs=x, variables=self.get_weight(vshape=[1, 1, self.channel, 16], name="encode1"), \
            stride_size=2, padding='SAME')
        encode1_bn = self.batch_norm(inputs=encode1, name="encode1_bn")
        encode1_act = tf.keras.activations.relu(encode1_bn)
        if(verbose): print(encode1_act.shape)
        self.conv_shapes.append(encode1_act.shape)

        encode2 = self.conv2d(inputs=encode1_act, variables=self.get_weight(vshape=[3, 3, 16, 32], name="encode2"), \
            stride_size=2, padding='SAME')
        encode2_bn = self.batch_norm(inputs=encode2, name="encode2_bn")
        encode2_act = tf.keras.activations.relu(encode2_bn)
        if(verbose): print(encode2_act.shape)
        self.conv_shapes.append(encode2_act.shape)

        encode3 = self.conv2d(inputs=encode2_act, variables=self.get_weight(vshape=[3, 3, 32, 64], name="encode3"), \
            stride_size=2, padding='SAME')
        encode3_bn = self.batch_norm(inputs=encode3, name="encode3_bn")
        encode3_act = tf.keras.activations.relu(encode3_bn)
        if(verbose): print(encode3_act.shape)

        return encode3_act

    def memory(self, z, verbose=False):

        if(verbose): print("Memory:", z.shape)
        N = 2000
        w_memory = self.get_weight(vshape=[1, 1, N, 64], bias=False, name="memory")
        cosim = self.cosine_sim(x1=z, x2=w_memory) # Eq.5
        atteniton = tf.nn.softmax(cosim) # Eq.4
        if(verbose): print(atteniton.shape)

        lam = 1 / N # deactivate the 1/N of N memories.

        addr_num = tf.keras.activations.relu(atteniton - lam) * atteniton
        addr_denum = tf.abs(atteniton - lam) + 1e-12
        memory_addr = addr_num / addr_denum
        if(verbose): print(memory_addr.shape)

        renorm = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))

        z_hat = tf.linalg.matmul(renorm, w_memory)
        if(verbose): print(z_hat.shape)

        return z_hat, renorm

    def decoder(self, z_hat, training=False, verbose=False):

        if(verbose): print("Decoder:", z_hat.shape)

        [n, h, w, c] = self.conv_shapes[-1]
        decode1 = self.conv2d_tr(inputs=z_hat, variables=self.get_weight(vshape=[3, 3, c, 64], transpose=True, name="decode1"), \
            output_shape=[tf.shape(z_hat)[0], 7, 7, 32], stride_size=2, padding='SAME')
        decode1_bn = self.batch_norm(inputs=decode1, name="decode1_bn")
        decode1_act = tf.keras.activations.relu(decode1_bn)
        if(verbose): print(decode1_act.shape)

        [n, h, w, c] = self.conv_shapes[-2]
        decode2 = self.conv2d_tr(inputs=decode1_act, variables=self.get_weight(vshape=[3, 3, c, 32], transpose=True, name="decode2"), \
            output_shape=[tf.shape(decode1_act)[0], h, w, c], stride_size=2, padding='SAME')
        decode2_bn = self.batch_norm(inputs=decode2, name="decode2_bn")
        decode2_act = tf.keras.activations.relu(decode2_bn)
        if(verbose): print(decode2_act.shape)

        [n, h, w, c] = self.conv_shapes[-3]
        decode3 = self.conv2d_tr(inputs=decode2_act, variables=self.get_weight(vshape=[3, 3, c, 16], transpose=True, name="decode3"), \
            output_shape=[tf.shape(decode2_act)[0], h, w, c], stride_size=2, padding='SAME')
        if(verbose): print(decode3.shape)

        x_hat = tf.clip_by_value(decode3, 1e-12, 1-(1e-12))

        return x_hat
