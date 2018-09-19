import tensorflow as tf
from model import RadarNet
from data_input_CIFAR10 import * 

NUM_CLASSES = 10 
save_model_path = './convnet_model_saved'
cifar10_datapath = '/home/jeonghwan/Desktop/CIFAR10_demo/cifar-10-batches-py'

def main():
    opt = {}
    opt['batch_size'] = 128
    opt['num_batch'] = 5
    opt['num_class'] = NUM_CLASSES
    opt['learning_rate'] = 1e-3
    opt['dropout_p'] = 0.5
    opt['epoch'] = 10
    opt['model_path'] = save_model_path 

    #Download the data if it is not present within the dir
    DownloadData(cifar10_datapath)

    batch_idx = 3
    sample_idx = 7001
    #Display overall stats
    display_stats(cifar10_datapath, batch_idx, sample_idx)

    # Preprocess all the data and save it
    preprocess_and_save_data(cifar10_datapath, normalize, one_hot_encode)

    #load the saved dataset
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='output')
    dropout_p = tf.placeholder(tf.float32, name='dropout_p')

    m = RadarNet(opt)
    model_out = m.model(x)
    cost = m.loss(model_out, y)
    optim = m.optimizer(cost)
    acc = m.accuracy(model_out, y)

    print("Training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(opt['epoch']):
            for i in range(1, opt['num_batch'] + 1):
               for batch_features, batch_labels in load_preprocess_training_batch(i, opt['batch_size']):
                    sess.run(optim, feed_dict={x: batch_features, y: batch_labels, dropout_p: opt['dropout_p']})
                    print('Batch #{}'.format(i))
                   # m.train_neural_network(sess, optim, opt['dropout_p'], batch_features, batch_labels)
               print('Epoch {:>2}, CIFAR10 Batch {}:'.format(epoch+1, i), end='')
               loss = sess.run(cost, feed_dict={x: batch_features, y: batch_labels, dropout_p: opt['dropout_p']})
               valid_acc = sess.run(acc, feed_dict={x: valid_features, y: valid_labels, dropout_p: opt['dropout_p']})
               print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
              # m.print_stats(sess, batch_features, batch_labels, cost, acc)

        saver = tf.train.Saver()
        save_path = saver.save(sess, opt['model_path'])


if __name__ == "__main__":
    main() 
