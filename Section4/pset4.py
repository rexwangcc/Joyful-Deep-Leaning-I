import tensorflow as tf
import numpy as np
import read_data
import model


def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('epoch {:d} iter {:d},  loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))

def main():
    # FOR CIFAR-10
    # dataset_generators = {
    #     'train': read_data.cifar10_dataset_generator('train', 512),
    #     'test': read_data.cifar10_dataset_generator('test', -1)
    # }
    
    # FOR SVHN
    dataset_generators = {
        'train': read_data.svhn_dataset_generator('train', 512),
        'test': read_data.svhn_dataset_generator('test', 512)
    }
    
    model_dict = model.apply_classification_loss(model.cnn_map)
    train_model(model_dict, dataset_generators, epoch_n=50, print_every=10)


main()
