import tensorflow as tf
import util as tf_utils
import  matplotlib ,argparse
import  matplotlib.pyplot as plt
import ir_model as ir
from erGAN import ERGAN

parser = argparse.ArgumentParser(description='Tensorflow erGAN Training...')

#  shadow model 
parser.add_argument('-b', '--batchSize', default='32', type=int)
parser.add_argument('-l', '--learnRate', default='0.0015', type=str)
args = parser.parse_args()

batchSize= args.batchSize 
learnRate= args.learnRate

config=  tf.ConfigProto(allow_soft_placement=True, log_device_placement=True )
config.gpu_options.allow_growth = True
sess   = tf.Session(config=config)
er_gan = ERGAN(sess=sess , batchSize=batchSize ,learnRate=learnRate )
er_gan.trian_model()