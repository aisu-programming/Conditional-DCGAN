''' Libraries '''
import os
import gc
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

from mapping import id2catergory


''' Functions '''
def make_ckpt_dir():
    ckpt_dir = f"ckpt/{time.strftime('%Y.%m.%d_%H.%M', time.localtime())}"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    return ckpt_dir


def load_data(catergory=None):
    (train_images, train_catergories), (test_images, test_catergories) = tf.keras.datasets.cifar10.load_data()
    if catergory is None:
        train_images = np.array(train_images, dtype=np.float32)
        test_images  = np.array(test_images, dtype=np.float32)
    else:
        train_images_tmp = []
        test_images_tmp  = []
        for i in range(len(train_images)):
            if train_catergories[i] == catergory:
                train_images_tmp.append(train_images[i])
        for i in range(len(test_images)):
            if test_catergories[i] == catergory:
                test_images_tmp.append(test_images[i])
        train_images = np.array(train_images_tmp, dtype=np.float32)
        test_images  = np.array(test_images_tmp, dtype=np.float32)
        train_catergories = np.array([[catergory]] * len(train_images), dtype=np.float32)
        test_catergories  = np.array([[catergory]] * len(test_images), dtype=np.float32)
    return train_images, train_catergories, test_images, test_catergories


def onehot_catergories(catergory_ids, real=None):
    label_onehots = np.eye(10)[np.squeeze(np.array(catergory_ids, dtype=np.int8))]
    if real is not None:
        if real: real_onehots = np.array([[1., 0.]] * len(np.squeeze(catergory_ids)))
        else: real_onehots = np.array([[0., 1.]] * len(np.squeeze(catergory_ids)))
        label_onehots = np.concatenate([real_onehots, label_onehots], axis=-1)
        # del real_onehots
        # gc.collect()
    return label_onehots


def plot_lr_schedule(ckpt_dir, lr_schedule):
    plt.figure(1, figsize=(12, 8), clear=True)
    plt.title('Learning Rate Schedule', fontsize=20)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.plot(lr_schedule(np.arange(60000, dtype=np.float32)))
    plt.savefig(f"{ckpt_dir}/lr_schedule.png", dpi=200)
    plt.close('all')
    return


def plot_history(ckpt_dir, history):
    
    gen_loss = history['gen_loss']
    dis_loss = history['dis_loss']
    lr       = history['lr']
    epochs_length = range(1, 1+len(gen_loss))

    plt.figure(1, figsize=(12, 8), clear=True)
    plt.suptitle('Losses & Learning Rate History', fontsize=20)
    plt.xlabel('Epochs')

    plt.subplot(1, 2, 1)
    plt.title('Losses')
    plt.plot(epochs_length, gen_loss, 'b-', label='Generator')
    plt.plot(epochs_length, dis_loss, 'r-', label='Discriminator')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title('Learning Rate')
    plt.plot(epochs_length, lr, 'r-')
    plt.legend()

    plt.savefig(f"{ckpt_dir}/history.png", dpi=200)
    plt.close('all')

    # del gen_loss, dis_loss, epochs_length
    # gc.collect()
    return


def plot_images(images, catergories, ckpt_dir, filename, title):

    images = np.array(images, dtype=np.uint8)

    plt.figure(1, figsize=(10, 10), clear=True)
    plt.suptitle(title, fontsize=20)
    for i in range(images.shape[0]):
        plt.subplot(4, 5, i+1)
        image = images[i, :, :, :]
        catergory = id2catergory[np.squeeze(catergories)[i]]
        plt.title(catergory)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/{filename}.png")
    plt.close('all')

    # del images, catergory
    # gc.collect()
    return


def plot_gif(ckpt_dir):
    gif_filename = f"{ckpt_dir}/advancement.gif"
    with imageio.get_writer(gif_filename, mode='I', fps=8) as writer:
        filenames = glob.glob(f"{ckpt_dir}/valid-*.png")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    # del gif_filename, filenames
    # gc.collect()
    return