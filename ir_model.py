import tensorflow.contrib.slim as slim
import tensorflow as tf
import  sys

'''
    In our work, we use Inception-Resnet model to help build a part of loss here.
    Also you can attempt to use other type of model to build loss or even withdrawn this 
    part of loss to save the Computing resources.
    We put this part here, as it indeed can imporve the performance.
'''
class Inp_ResNet():
    def __init__(self,num_of_labs):
        self.num_of_labs= num_of_labs
        self.is_train=True
    def build_get_loss(self,imgs , labs ):
        self.imgs=imgs
        self.labs = labs
        inputs =tf.reshape(imgs ,(imgs.shape[0] ,imgs.shape[1],imgs.shape[2],imgs.shape[3])   ) 
        x = self._stem_init(inputs)
        for i in range(5):
            x=self._incep_res_A(x, scale=0.17 ,ind=i)
        x=self._reduction_a(x,192  , 192  , 256 , 384  )

        for i in range(10):
            x=self._incep_res_B(x , scale= 0.10 , ind =i  )

        x=self._reduction_b(x )
        for i in range(5):
            x=self._incep_res_C(x , scale = 0.20,ind = i )
        embedding= self._avgP_drop_soft(x,self.num_of_labs)
        self.emb =embedding
        return  embedding
        
    def _loss_AMSoftmax(self ,embeddings, nrof_classes):

        m = 0.35
        s = 50
        embeddings= slim.dropout(embeddings, 0.8 , is_training=self.is_train ,
                                       scope='Dropout')
        with tf.variable_scope('AM_logits' , reuse=tf.AUTO_REUSE):
            print(embeddings.shape)
            kernel = tf.get_variable(name='kernel',dtype=tf.float32,shape=[embeddings.shape[-1],nrof_classes],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
            embe = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
            cos_theta = tf.matmul(embe, kernel_norm)

            cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
            phi = cos_theta - m

            self.labs= tf.one_hot(self.labs,nrof_classes ,on_value=1,off_value=None,axis=1)
            self.labs= tf.squeeze(self.labs)
            print('aaaaaaa', self.labs.shape , phi.shape , cos_theta.shape )

            adjust_theta = s * tf.where(tf.equal(self.labs,1), phi, cos_theta)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labs, logits=adjust_theta, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            return cross_entropy_mean

    def _avgP_drop_soft(self ,net ,bottleneck_layer_size):
        with tf.variable_scope('Logits' ,reuse=tf.AUTO_REUSE):
             net = slim.conv2d(net, 1792, 7, stride=1, padding='VALID', scope='AvgPool_1a_8x8')
             net= self._BN_tool(net,'p1')
             embdd = slim.flatten(net)
             return  embdd

    # Inception-Resnet-C 
    def _incep_res_C(self ,net, scale=1.0  , activation_fn=tf.nn.relu, ind = 0,reuse=tf.AUTO_REUSE):
        """Builds the 8x8 resnet block."""
        with tf.variable_scope('Block8' + str(ind),reuse=reuse):
            with tf.variable_scope('Branch_0'):
                # 8*8*192
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
                tower_conv= self._BN_tool(tower_conv,'y1')

            with tf.variable_scope('Branch_1'):
                # 8*8*192
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_0= self._BN_tool(tower_conv1_0,'y2')
                # 8*8*192
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                            scope='Conv2d_0b_1x3')
                tower_conv1_1= self._BN_tool(tower_conv1_1,'y3')
                # 8*8*192
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                            scope='Conv2d_0c_3x1')
                tower_conv1_2= self._BN_tool(tower_conv1_2,'y4')
            # 8*8*384
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)
            # 8*8*1792
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
            up= self._BN_tool(up,'y5')
            # scale=0.20
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net

    # # Reduction-B   
    def _reduction_b(self ,net):
        with tf.variable_scope('red_b_Branch_0' ,reuse=tf.AUTO_REUSE):
            # 17*17*256
            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv = self._BN_tool(tower_conv,'t1')
            # 8*8*384
            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')

            tower_conv_1= self._BN_tool(tower_conv_1,'t2')
        with tf.variable_scope('red_b_Branch_1',reuse=tf.AUTO_REUSE):
            # 17*17*256
            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')

            tower_conv1= self._BN_tool(tower_conv1,'t3')

            # 8*8*256
            tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')

            tower_conv1_1= self._BN_tool(tower_conv1_1,'t4')

        with tf.variable_scope('red_b_Branch_2',reuse=tf.AUTO_REUSE):
            # 17*17*256
            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2= self._BN_tool(tower_conv2,'t5')
            # 17*17*256
            tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_1= self._BN_tool(tower_conv2_1,'t6')
            # 8*8*256
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')
            tower_conv2_2= self._BN_tool(tower_conv2_2,'t7')
        with tf.variable_scope('red_b_Branch_3' ,reuse=tf.AUTO_REUSE):
            # 8*8*896
            # tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
            #                              scope='MaxPool_1a_3x3')
            # (80, 16, 16, 896)   (80, 7, 7, 896)
            tower_pool = slim.conv2d(net, 896, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            tower_pool= self._BN_tool(tower_pool,'p2')

        # 8*8*1792
        net = tf.concat([tower_conv_1, tower_conv1_1,
                            tower_conv2_2, tower_pool], 3)
        return net

    # Inception-Renset-B 
    def _incep_res_B(self ,net, scale=1.0 , activation_fn=tf.nn.relu, ind=0 , reuse=tf.AUTO_REUSE):
        """Builds the 17x17 resnet block."""
        with tf.variable_scope('Block17' + str(ind),  reuse=reuse):
            with tf.variable_scope('Branch_0'):
                # 17*17*128
                tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
                tower_conv = self._BN_tool(tower_conv,'r1')
            with tf.variable_scope('Branch_1'):
                # 17*17*128
                tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
                tower_conv1_0 = self._BN_tool(tower_conv1_0,'r2')
                # 17*17*128
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                            scope='Conv2d_0b_1x7')
                tower_conv1_1 = self._BN_tool(tower_conv1_1,'r3')
                # 17*17*128
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                            scope='Conv2d_0c_7x1')
                tower_conv1_2 = self._BN_tool(tower_conv1_2,'r4')
            # 17*17*256
            mixed = tf.concat([tower_conv, tower_conv1_2], 3)
            # 17*17*896
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
            up = self._BN_tool(up,'r5')
            net += scale * up
            if activation_fn:
                # net=self._BN_tool(net,'f_'+str(ind))
                net = activation_fn(net)
        return net

    # Reduction-A 
    def _reduction_a(self ,net, k, l, m, n):
        with tf.variable_scope('Branch_0' , reuse=tf.AUTO_REUSE):
            tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
            tower_conv = self._BN_tool(tower_conv,'e1')
        with tf.variable_scope('Branch_1' , reuse=tf.AUTO_REUSE):
            tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
            tower_conv1_0 = self._BN_tool(tower_conv1_0,'e2')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv1_1 = self._BN_tool(tower_conv1_1,'e3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                        stride=2, padding='VALID',
                                        scope='Conv2d_1a_3x3')
            tower_conv1_2 = self._BN_tool(tower_conv1_2,'e4')
        with tf.variable_scope('Branch_2' , reuse=tf.AUTO_REUSE ):
            tower_pool = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            tower_pool= self._BN_tool(tower_pool,'p3')

        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
        return net
    # Inception-Renset-A 
    def _incep_res_A(self ,net, scale=1.0  , activation_fn=tf.nn.relu, ind=0, reuse=tf.AUTO_REUSE):
        """Builds the 35x35 resnet block."""
        with tf.variable_scope('Block35'+ str(ind),  reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
                tower_conv = self._BN_tool(tower_conv,'w1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_0 = self._BN_tool(tower_conv1_0,'w2')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
                tower_conv1_1 = self._BN_tool(tower_conv1_1,'w3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_0 = self._BN_tool(tower_conv2_0,'w4')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
                tower_conv2_1 = self._BN_tool(tower_conv2_1,'w5')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
                tower_conv2_2 = self._BN_tool(tower_conv2_2,'w6')

            mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,activation_fn=None, scope='Conv2d_1x1')
            up = self._BN_tool(up,'w7')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net
    def _stem_init(self ,inputs):
        with tf.variable_scope('Stem_Block', reuse=tf.AUTO_REUSE):
            net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
            net = self._BN_tool(net,'q1')
            print('1 ',len(tf.global_variables()))

            net = slim.conv2d(net, 32, 3, padding='VALID',
                        scope='Conv2d_2a_3x3')
            net = self._BN_tool(net,'q2')
            net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
            net = self._BN_tool(net,'q3')
            net = slim.conv2d(net, 64, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
            net= self._BN_tool(net,'p4')

            net = slim.conv2d(net, 80, 1, padding='VALID',
                        scope='Conv2d_3b_1x1')
            net = self._BN_tool(net,'q5')
            net = slim.conv2d(net, 192, 3, padding='VALID',
                    scope='Conv2d_4a_3x3')
            net = self._BN_tool(net,'q6')
            net = slim.conv2d(net, 256, 3, stride=1, padding='VALID',
                    scope='Conv2d_4b_3x3')
            net = self._BN_tool(net,'q7')

            return  net

    def _weight_decay(self ):
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find(r'weights') > 0:
            costs.append(tf.nn.l2_loss(var))

        return tf.multiply( 0.0002, tf.add_n(costs))
    def _BN_tool(self , x, name):
        with tf.variable_scope(name ,reuse=tf.AUTO_REUSE) as scope:
            return tf.contrib.layers.batch_norm(  x,
                                               decay=0.95,
                                               epsilon=0.001,
                                               scope=scope,
                                               is_training=self.is_train,
                                               fused=True,
                                               updates_collections=None)

