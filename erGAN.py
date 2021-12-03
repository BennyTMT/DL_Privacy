import tensorflow as tf
import util as tf_utils
import  random
import  numpy as np
import  sys
import  os
import  matplotlib
import  matplotlib.pyplot as plt
import ir_model as ir
IND_GPU =1
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
class ERGAN():
    def __init__(self, sess=None , batchSize=32 , learnRate=0.0015):
        
        self.batch_size = batchSize
        self.sess       = sess
        self.learnRate  = learnRate

        # fixed parameters, also one can adjust it slightly
        self.picStorePath = 'faceRecovered/'
        self.dataPath     = 'lfwDataSet/'
        self.modelSaved   = 'erGAN/'
        
        self.dis_c = 32
        self.cos_lambda = 1
        self.L1_lambda  = 30
        self.g_lambda   = 10
        self.Pr_lambda  = 1
        self._gen_train_ops, self._dis_train_ops = [], []
        self.is_train= True
        self.name_idx = 0 
        self._build_net()
        print('Initialized erGAN SUCCESS!\n')

    def _build_net(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 160,160, 3], name='image')
        self.emb_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size,1792], name='emb_place')

        self.lrt_d =tf.placeholder(tf.float32,shape=[] , name='lrt_d')
        self.lrt_g =tf.placeholder(tf.float32,shape=[] , name='lrt_g')

        '''
            Generate Face Image from Embedding 
            based on this Face Image to build a part of the loss:
                It seems a little complicated, but we found that our model can get good performance in 
                recover task without this part of loss. We still put it here to make the whole task
                completed, as it can get Slight Improvement indeed.
        '''
        self.fake_X    =   self._generator(self.emb_placeholder)
        self.model =ir.Inp_ResNet(10575)
        y =tf.placeholder(tf.int32,shape=[None,] )
        fake_emb = self.model.build_get_loss(self.fake_X ,y)
        # =============================================================
        # Set discriminator\generator Loss 
        # =============================================================
        _vars = tf.all_variables() 
        self.variable_to_save = [var for var  in _vars if 'g_' not  in var.name]
        d_real, d_logit_real = self.discriminator(self.X)
        d_fake, d_logit_fake = self.discriminator(self.fake_X , is_reuse=True)
        # you also can use d_real&d_fake to build the loss, but we find current loss get better performance.
        self.d_1  =  tf.reduce_mean(d_logit_fake ) 
        self.d_2  =  tf.reduce_mean(d_logit_real)
        self.d_loss = self.d_1 - self.d_2
        self.gan_loss  =self.g_lambda * (- self.d_1 )
        

        # Gradient Penalty########################################################################################################3
        # This trick is well-known and Necessary 
        self.epsilon = tf.random_uniform(
                                shape=[self.batch_size, 1, 1, 1],
                                minval=0.,
                                maxval=1.)
        X_hat = self.X + self.epsilon * (self.fake_X - self.X)

        D_X_hat = self.discriminator(X_hat, is_reuse=True)
        grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]

        slopes = tf.sqrt(1e-8 +  tf.reduce_sum(tf.square(grad_D_X_hat),  tf.range(1, len(grad_D_X_hat.get_shape())) ))

        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.d_loss = self.d_loss + 10.0 * gradient_penalty

        self.cos_dis  =  self._count_cos(self.emb_placeholder , fake_emb)
        l1_dis        =  tf.reduce_mean(tf.abs(self.X -self.fake_X ))
        self.l1_loss  =  self.L1_lambda * l1_dis
        self.g_loss   =  self.gan_loss + self.l1_loss - self.cos_lambda * self.cos_dis
        # Finish the whole loss set===============================================

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discr_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            print('Build opt ........')
            self.dis_op = tf.train.AdamOptimizer(learning_rate=self.lrt_d, name='d_',beta1=0, beta2=0.9 ,epsilon=1e-08).minimize(self.d_loss, var_list=d_vars)
            self.gen_op = tf.train.AdamOptimizer(learning_rate=self.lrt_d, name='g_',beta1=0, beta2=0.9 ,epsilon=1e-08).minimize(self.g_loss, var_list=g_vars)
        
        print('Loaded Model!!!!!!!')
    def _generator(self ,emb,name='g_'):
        # You can adjust this hyper parameters,but it could increase the Memory consumption
        # This generator is for the task who`s embedding size is 1024
        self.std_1=[1,1,1,1]
        self.std_2=[1,2,2,1]
        self.std_3=[1,4,4,1]
        self.std_4=[1,10,10,1]
        self.std_5=[1,5,5,1]
        self.N = 6

        input_emb = emb
        with tf.variable_scope(name , reuse=tf.AUTO_REUSE ) :
            input=tf.reshape(input_emb,(self.batch_size , 1,1,int(input_emb.shape[1])))
            input=self._BN_norm(input)
            
            r1 = 0.75 ; r2 =0.25  

            x_1=self._construct_de_conv(input, 1024 ,[int (input.get_shape()[0])
                                      ,10,10,1024 ] ,self.std_4)
            x_1=self._BN_norm(x_1)
            x_1=self._unline_f(x_1)
            x_1=self._construct_de_conv(x_1, 256 ,[int (input.get_shape()[0])
                                      ,10,10,256 ] ,self.std_1)
            x_1=self._BN_norm(x_1)
            x_1=tf.nn.relu(x_1)
            x_1=self._unline_f(x_1)
            #------------------------------------------------------------------------------
            x_2=self._construct_de_conv(input, 1024 ,[int (input.get_shape()[0])
                                      ,5,5, 1024 ] ,self.std_5)
            x_2=self._BN_norm(x_2)
            x_2=self._unline_f(x_2)
            x_2=self._construct_de_conv(x_2, 256 ,[int (input.get_shape()[0])
                                      ,10,10,256  ] ,self.std_2)
            x_2=self._BN_norm(x_2)
            x_2=self._unline_f(x_2)
            #------------------------------------------------------------------------------
            x_3=self._construct_de_conv(input, 1024 ,[int (input.get_shape()[0])
                                      ,2,2,1024 ] ,self.std_2)
            x_3=self._BN_norm(x_3)
            x_3=self._unline_f(x_3)
            x_3=self._construct_de_conv(x_3, 512 ,[int (input.get_shape()[0])
                                      ,10,10, 512 ] ,self.std_5)
            x_3=self._BN_norm(x_3)
            x_3=self._unline_f(x_3)
            #------------------------------------------------------------------------------
            # concatenate all the infomation 
            x= tf.concat([x_1, x_2 , x_3] ,axis = -1  )
            x=self._construct_de_conv(x ,int(  x.get_shape()[3])  //2  , [int (x.get_shape()[0])
                                      ,20,20,int ( x.get_shape()[3] ) //2  ] , self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t =x 
            # Here is a residual block
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=tf.nn.relu(x)
            x_20_1= r1 * t+ r2 * x
            
            x=self._construct_de_conv(input , 1024     , [int (input.get_shape()[0])
                                      ,2,2, 1024   ] , self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)

            # You can change layer size here 
            t =x
            for _ in range(2):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x= r1 * t+ r2 * x

            #-----------------------------------------------------------------------------------
            x=self._construct_de_conv(x , 1024   , [int (x.get_shape()[0])
                                      ,4,4,1024  ] , self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t =x
            for i in range(2):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x= r1 * t+ r2 * x
 
            #--------------------------------------------------------------------------------
            x=self._construct_de_conv(x , 512   , [int (x.get_shape()[0])
                                      ,20,20, 512  ] , self.std_5)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t =x
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x_20_2= r1 * t+ r2 * x
            x = tf.concat([x_20_1, x_20_2] , axis = -1 )

            x=self._construct_de_conv(x,int(  x.get_shape()[3]) //2,[int (x.get_shape()[0])
                                      ,40,40,int ( x.get_shape()[3] ) //2 ] , self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t =x 
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x = r1 * t+ r2 * x
            #-------------------------------------------------------------------------------------------------
            x=self._construct_de_conv(x,int(  x.get_shape()[3]) //2,[int (x.get_shape()[0])
                                      ,40,40,int ( x.get_shape()[3] ) //2 ] , self.std_1)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t =x 
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)

            x = r1 * t+ r2 * x 
            #-------------------------------------------------------------------------------------------------
            x=self._construct_de_conv(x,int(  x.get_shape()[3]) //2, [int (x.get_shape()[0])
                                      ,80,80,int ( x.get_shape()[3] ) //2 ], self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t = x            
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)

            x = r1 * t+ r2 * x
            #-----------------------------------------------------------------------------------------------------
            x=self._construct_de_conv(x,int(  x.get_shape()[3]) //4, [int (x.get_shape()[0])
                                      ,80,80,int ( x.get_shape()[3] ) //4 ], self.std_1)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t = x 
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x = r1 * t+ r2 * x 
            #--------------------------------------------------------------------------------------------------------
            x=self._construct_de_conv(x,int(  x.get_shape()[3]) // 2, [int(x.get_shape()[0]),
                    160 , 160,int(x.get_shape()[3])//2 ] , self.std_2)
            x=self._BN_norm(x)
            x=self._unline_f(x)

            t= x 
            print('5' , x.shape)
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)

            x = r1 * t+ r2 * x 
            #---------------------------------------------------------------------------------------------------------
            x=self._construct_de_conv(x,int(  x.get_shape()[3]) // 4, [int(x.get_shape()[0]),
                    160 , 160,int(x.get_shape()[3])//4 ] , self.std_1)
            x=self._BN_norm(x)
            x=self._unline_f(x)
            t= x
            for i in range(self.N):
                x=self._construct_de_conv(x,int(  x.get_shape()[3]) ,self._get_out_size(x,2) , self.std_1)
                x=self._BN_norm(x)
                x=self._unline_f(x)
            x = r1 * t+ r2 * x
            #---------------------------------------------------------------------------------------------------------
            out_put=self._construct_de_conv(x,3, [int(x.get_shape()[0]),
                    int(x.get_shape()[1]) , int(x.get_shape()[2]),3], self.std_1)
            print('6' , out_put.shape)
            out_put=self._BN_norm(out_put)
            out_put=tf.nn.tanh(out_put)

        return  ( out_put+1 ) * 127.5
    def discriminator(self, data, is_reuse=False  , discType='image'):
        # In this project, we use "image"
        if  discType== 'pixel':
            return self.discriminator_pixel(data, is_reuse=is_reuse)
        elif discType == 'image':
            return self.discriminator_image(data, is_reuse=is_reuse)
        else:
            raise NotImplementedError

    def discriminator_pixel(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()

            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
            conv1 = tf_utils.lrelu(conv1, name='conv1_lrelu1')

            conv2 = tf_utils.conv2d(conv1, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
            conv2 = tf_utils.lrelu(conv2)

            conv3 = tf_utils.conv2d(conv2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf_utils.lrelu(conv3)

            output = tf_utils.conv2d(conv3, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

            return tf.nn.sigmoid(output), output

    def discriminator_image(self, data, name='discr_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True: scope.reuse_variables()

            N =1
            conv1 = tf_utils.conv2d(data, self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv1_conv1')
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')

            for z in range(N):
                conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2_' + str(z) )
                conv1 = tf.nn.relu(conv1, name='conv1_relu2_' + str(z))

            conv1 = tf_utils.conv2d(conv1, self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')

            pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

            # conv2: (N, 160, 160, 32) -> (N, 40, 40, 64)
            conv2 = tf_utils.conv2d(pool1, 2*self.dis_c, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2_conv1')
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            
            for z in range(N):
                conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2_' + str(z))
                conv2 = tf.nn.relu(conv2, name='conv2_relu2_' + str(z))

            conv2 = tf_utils.conv2d(conv2, 2*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

            conv3 = tf_utils.conv2d(pool2, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = tf_utils.conv2d(conv3, 4*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

            conv4 = tf_utils.conv2d(pool3, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = tf_utils.conv2d(conv4, 8*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

            conv5 = tf_utils.conv2d(pool4, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = tf_utils.conv2d(conv5, 16*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')
 
            conv6 = tf_utils.conv2d(conv5, 32*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv1')
            conv6 = tf.nn.relu(conv6, name='conv6_relu1')
            conv6 = tf_utils.conv2d(conv6, 32*self.dis_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv2')
            conv6 = tf.nn.relu(conv6, name='conv6_relu2')

            shape = conv6.get_shape().as_list()
            gap = tf.layers.average_pooling2d(inputs=conv5, pool_size=shape[1], strides=1, padding='VALID',
                                              name='global_vaerage_pool')
            gap_flatten = tf.reshape(gap, [-1, 16*self.dis_c])
            output = tf_utils.linear(gap_flatten, 1, name='linear_output')

            return tf.nn.sigmoid(output), output

    # Functions related to Build Model ======================= top
    def _count_cos(self,f1 , f2 ):
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(f1), axis=1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(f2), axis=1))
        x1_x2 = tf.reduce_sum(tf.multiply(f1, f2), axis=1)
        cosin = x1_x2 / (x1_norm * x2_norm)
        return  tf.reduce_mean(cosin)

    def _count_pearson(  self , fla_1, fla_2  ):

        f1 =fla_1- tf.expand_dims(tf.reduce_mean(fla_1,axis=1) ,axis=1)
        f2 =fla_2- tf.expand_dims(tf.reduce_mean(fla_2,axis=1) ,axis=1)
        cov=tf.reduce_sum(f1 *f2 , axis=1)
        voc=tf.sqrt(tf.reduce_sum( f1**2  ,axis=1)) * tf.sqrt(tf.reduce_sum( f2**2  ,axis=1))
        return  tf.reduce_mean(cov/voc)

    def _unline_f(self , x):
        return  tf_utils.lrelu(x , 0.01 )
        
    def _layer_norm(self ,   x, trainable=True, name='layer_norm'):
        with tf.variable_scope(name):
            return tf.contrib.layers.layer_norm(x, trainable=trainable)

    def _construct_de_conv(self ,input , out_filter ,out_shape ,stride ):
        in_filter = int ( input.get_shape()[3] )
        name = str( self.name_idx )
        self.name_idx  += 1 
        #name = str( random.randint(1,1000000) )

        with tf.variable_scope(name) :
            _filter=tf.get_variable('weights', [3,3,out_filter,in_filter] ,tf.float32 ,
                                            initializer=tf.random_normal_initializer(stddev=0.0001))

        return  tf.nn.conv2d_transpose(input,filter=_filter,output_shape=out_shape,strides=stride , padding='SAME')

    def _BN_norm(self , x):
        #name = str( random.randint(1,1000000) )
        name = str(self.name_idx  ) 
        self.name_idx +=1 
        with tf.variable_scope(name) :
            return tf.contrib.layers.batch_norm(  x,
                                                decay=0.9,
                                                epsilon=0.001,
                                                scope='un_conv_bn',
                                                is_training=self.is_train,
                                                fused=True,
                                                updates_collections=None)

    def _get_out_size(self , input  , k ):
        if k ==1 : return  [int (input.get_shape()[0] ) , int (input.get_shape()[1]) * 2 , int(input.get_shape()[2]) * 2 ,int(input.get_shape()[3]) //2 ]
        else:  return  [int(input.get_shape()[0]),int(input.get_shape()[1]) , int(input.get_shape()[2])
                                    ,int(input.get_shape()[3])]
    # Functions related to Build Model ======================= bottom

    # Train the model 
    def trian_model(self):
        print('begin to trian ..................')
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver(self.variable_to_save)
        # one can change your own path here to load your model
        saver.restore(self.sess, "model_std_ir_drop_tmp/model.ckpt")
        
        # you can save the whole model, or part of it 
        # ll = list(set(tf.all_variables()) -  set(self.variable_to_save )  )
        ll = tf.all_variables() 
        saver = tf.train.Saver(ll )

        # Init...
        lrt = self.lear_rat  ; l1  = 0; dl =1 ;gl =1 ;epoch=200 ;sub_train_times=1
        for  i in range(epoch):
            self.is_train= True
            lrt = lrt * 0.965
            for ind in range(255):
                img_np =self.dataPath + 'batch_'+str(ind)+'_img.npy'
                emb_np =self.dataPath + 'batch_'+str(ind)+'_emb.npy'
                data  =np.load(img_np)
                embd  =np.load(emb_np)
                for _ in  range(sub_train_times):
                    # one can change random range based on your batch size 
                    locate=random.randint(0 , 18)
                    data =data[locate : locate + self.batch_size]
                    embd =embd[locate : locate + self.batch_size]
                    for _ in range(1):
                        self.sess.run(self.dis_op,feed_dict={self.X : data , self.lrt_d:lrt,self.emb_placeholder:embd})
                    for _ in  range(5):
                        if i < 20:
                            self.sess.run(self.gen_op,feed_dict={self.X : data , self.lrt_d:lrt,self.emb_placeholder:embd})
                        else :
                            self.sess.run(self.gen_op,feed_dict={self.X : data , self.lrt_d:lrt,self.emb_placeholder:embd})
                if ind % 50 == 0 or ind ==1   :
                    l1 , ga , dl , gl , cos_d =self.sess.run([ self.l1_loss , self.gan_loss, self.d_loss, self.g_loss , self.cos_dis ], feed_dict={
                        self.X : data,
                        self.emb_placeholder:embd 
                    }) 
                    print(' self.l1_loss %.4f self.gan_loss %.4f , self.d_loss  %.4f , self.g_loss %.4f , self.cos_dis %.4f  ' %(l1,ga,dl,gl,cos_d))

            if i % 3 == 0 :
                self.is_train= False 
                if l1 > 200:
                    saver.save(self.sess, self.modelSaved + 'model_tmp/model.ckpt')
                else :
                    saver.save(self.sess, self.modelSaved + 'model_new/model.ckpt') 

                # randomly chose a picture to plt the fake Image 
                # which can help to monitor the performance 
                ind = random.randint(1,200)
                img_np =self.dataPath+'batch_'+str(ind)+'_img.npy'
                emb_np =self.dataPath+'batch_'+str(ind)+'_emb.npy'

                data   =np.load(img_np) 
                embd   =np.load(emb_np)

                # batch size 
                data =  data[:self.batch_size]
                embd =  embd[:self.batch_size]

                g_imgs= self.sess.run([self.fake_X] , feed_dict={
                    self.X : data, self.emb_placeholder:embd
                })
                g_img =np.squeeze(np.array(g_imgs)) 

                print('Saving the face image　........')       
                # randomly chose some imgs we recovered to save 
                s_in = random.randint(0, 25)
                path = self.picStorePath + 'save_img_'+str(IND_GPU)+'/'+str(i) + '_g.png'
                matplotlib.image.imsave(path, g_img[s_in,...].astype(np.int32))

                path = self.picStorePath + 'save_img_'+str(IND_GPU)+ '/'+str(i) + '_r.png'
                matplotlib.image.imsave(path, data[s_in,...])

                s_in = random.randint(0, 25)
                path = self.picStorePath +'save_img_'+str(IND_GPU)+'/'+str(i+1) + '_g.png'
                matplotlib.image.imsave(path, g_img[s_in,...].astype(np.int32))

                path = self.picStorePath +'save_img_'+str(IND_GPU)+'/'+str(i+1) + '_r.png'
                matplotlib.image.imsave(path, data[s_in,...])
