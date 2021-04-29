''' Libraries '''
import gc
import math
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from mapping import catergory2id
from utils import make_ckpt_dir, load_data, onehot_catergories, plot_lr_schedule, plot_history, plot_images, plot_gif
from generator import generator_loss, MyGenerator
from discriminator import discriminator_loss, MyDiscriminator


''' Parameters '''
CKPT_DIR = make_ckpt_dir()
# RESTORE_CKPT_PATH = r"ckpt/2021.04.28_11.34/latest_ckpt/ckpt-18"
# LOAD_PARAMETERS_PATH = r"ckpt/2021.04.28_11.34/parameters.txt"
# START_EPOCH = 181
RESTORE_CKPT_PATH = None
LOAD_PARAMETERS_PATH = None
START_EPOCH = 0
CATERGORY = 'all'
BATCH_SIZE = 16
NOISE_DIM = 1024
G_LAYER_NUM = 4
D_LAYER_NUM = 6
DROPOUT = 0.4
EPOCH = 400
INITIAL_LR = 2.6e-4
WARMUP_STEPS = 7500
DECAY_STEPS = 500
DECAY_RATE = 0.99
G_D_LIMIT = None
G_D_RATIO = None


''' Functions '''
def save_parameters():
    with open(f"{CKPT_DIR}/parameters.txt", mode='w') as f:
        f.write(f"RESTORE_CKPT_PATH   : {RESTORE_CKPT_PATH}\n")
        f.write(f"LOAD_PARAMETERS_PATH: {LOAD_PARAMETERS_PATH}\n")
        f.write(f"START_EPOCH         : {START_EPOCH}\n")
        f.write(f"CATERGORY           : {CATERGORY}\n")
        f.write(f"BATCH_SIZE          : {BATCH_SIZE}\n")
        f.write(f"NOISE_DIM           : {NOISE_DIM}\n")
        f.write(f"G_LAYER_NUM         : {G_LAYER_NUM}\n")
        f.write(f"D_LAYER_NUM         : {D_LAYER_NUM}\n")
        f.write(f"DROPOUT             : {DROPOUT}\n")
        f.write(f"EPOCH               : {EPOCH}\n")
        f.write(f"INITIAL_LR          : {INITIAL_LR}\n")
        f.write(f"WARMUP_STEPS        : {WARMUP_STEPS}\n")
        f.write(f"DECAY_STEPS         : {DECAY_STEPS}\n")
        f.write(f"DECAY_RATE          : {DECAY_RATE}\n")
        f.write(f"G_D_LIMIT           : {G_D_LIMIT}\n")
        f.write(f"G_D_RATIO           : {G_D_RATIO}\n")
        return


def load_parameters():
    raise NotImplementedError
    with open(f"{LOAD_PARAMETERS_PATH}", mode='r') as f:

        f.readline()
        f.readline()
        f.readline()
        
        CATERGORY   = str(f.readline().split(':')[1].strip())
        BATCH_SIZE  = int(f.readline().split(':')[1].strip())
        NOISE_DIM   = int(f.readline().split(':')[1].strip())
        DROPOUT     = float(f.readline().split(':')[1].strip())
        EPOCH       = int(f.readline().split(':')[1].strip())
        INITIAL_LR  = float(f.readline().split(':')[1].strip())
        DECAY_STEPS = int(f.readline().split(':')[1].strip())
        DECAY_RATE  = float(f.readline().split(':')[1].strip())

        G_D_LIMIT   = str(f.readline().split(':')[1].strip())
        if G_D_LIMIT == 'None': G_D_LIMIT = None
        else: G_D_LIMIT = int(G_D_LIMIT)

        G_D_RATIO   = str(f.readline().split(':')[1].strip())
        if G_D_RATIO == 'None': G_D_RATIO = None
        else: G_D_RATIO = int(G_D_RATIO)

        return


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate):
        super(CustomSchedule, self).__init__()
        self.ilr = initial_lr
        self.ws = warmup_steps
        self.ds = decay_steps
        self.dr = decay_rate
    def __call__(self, step):
        arg1 = step / self.ws
        arg2 = self.dr ** ((step - self.ws) / self.ds)
        return self.ilr * tf.math.minimum(arg1, arg2)


@tf.function
def train_generator(f_noise, f_ctg):
    with tf.GradientTape() as gen_tape:
        f_img_tmp = generator(f_noise[BATCH_SIZE:], f_ctg[BATCH_SIZE:])
        f_predict = discriminator(f_img_tmp, f_ctg[BATCH_SIZE:])
        gen_loss  = generator_loss(f_predict)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    train_gen_loss(gen_loss)
    return


@tf.function
def train_discriminator(r_img, r_ctg, f_noise, f_ctg):
    with tf.GradientTape() as dis_tape:
        r_predict = discriminator(r_img, r_ctg, training=True)
        f_img_tmp = generator(f_noise[:BATCH_SIZE], f_ctg[:BATCH_SIZE])
        f_predict = discriminator(f_img_tmp, f_ctg[:BATCH_SIZE], training=True)
        dis_loss = discriminator_loss(r_predict, f_predict)
    dis_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
    train_dis_loss(dis_loss)
    return


def train_step(epoch, batch_idx, r_img, r_ctg, f_noise, f_ctg):

    general_rule = G_D_RATIO is None or train_gen_loss.result() < 1e-5 or train_dis_loss.result() < 1e-5
    gen_rule = G_D_LIMIT is None or not ((epoch+1) % G_D_LIMIT == 0 and train_dis_loss.result() / train_gen_loss.result() > G_D_RATIO)
    # gen_rule = batch_idx < 50 or True
    dis_rule = G_D_LIMIT is None or not ((epoch+1) % G_D_LIMIT == 0 and train_gen_loss.result() / train_dis_loss.result() > G_D_RATIO)
    # dis_rule = batch_idx < 50 or (batch_idx % 8) == 0

    if general_rule or gen_rule: train_generator(f_noise, f_ctg)
    if general_rule or dis_rule: train_discriminator(r_img, r_ctg, f_noise, f_ctg)

    return


def main():

    global generator, discriminator
    generator = MyGenerator(G_LAYER_NUM)
    discriminator = MyDiscriminator(D_LAYER_NUM, DROPOUT)

    global optimizer
    lr_schedule = CustomSchedule(
        initial_lr=INITIAL_LR,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    plot_lr_schedule(CKPT_DIR, lr_schedule)

    # Checkpoint
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        generator=generator,
        discriminator=discriminator
    )
    if RESTORE_CKPT_PATH is not None and LOAD_PARAMETERS_PATH is not None:
        checkpoint.restore(save_path=RESTORE_CKPT_PATH)
        load_parameters()
    manager = tf.train.CheckpointManager(checkpoint, directory=f"{CKPT_DIR}/latest_ckpt", max_to_keep=1)
    status = checkpoint.restore(manager.latest_checkpoint)

    save_parameters()

    if CATERGORY == 'all': real_images, real_catergories, _, _ = load_data()
    else: real_images, real_catergories, _, _ = load_data(catergory2id[CATERGORY])
    # plot_images(real_images[:20], real_catergories[:20], CKPT_DIR,
    #             filename="demo", title="Train Images Demonstration")

    global train_gen_loss, train_dis_loss, valid_gen_loss, valid_dis_loss
    train_gen_loss = tf.keras.metrics.Mean()
    train_dis_loss = tf.keras.metrics.Mean()

    gen_losses_history = []
    dis_losses_history = []
    lr_history = []

    # Validation
    valid_noise = np.random.normal(size=(20, NOISE_DIM))
    if CATERGORY == 'all':
        valid_ctg = np.array([
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
            5, 5, 6, 6, 7, 7, 8, 8, 9, 9,
        ])
    else:
        valid_ctg = np.array([[catergory2id[CATERGORY]]] * 20)

    for epoch in range(START_EPOCH, EPOCH):

        r_imgs, r_ctgs = unison_shuffled_copies(real_images, real_catergories)

        r_imgs = r_imgs[:8000]
        r_ctgs = r_ctgs[:8000]

        train_gen_loss.reset_states()
        train_dis_loss.reset_states()

        progress_bar = tqdm(tf.range(math.ceil(len(r_imgs)/BATCH_SIZE)),
                            total=math.ceil(len(r_imgs)/BATCH_SIZE),
                            ascii=True, desc=f"Epoch: {epoch+1:2d}")
        for i in progress_bar:

            r_img_batch = r_imgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            r_ctg_batch = r_ctgs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            f_noise = np.random.normal(size=(BATCH_SIZE*2, NOISE_DIM))
            if CATERGORY == 'all':
                f_ctg = np.random.randint(0, 10, size=BATCH_SIZE*2)
            else:
                f_ctg = np.array([[catergory2id[CATERGORY]]] * BATCH_SIZE*2)

            train_step(epoch, int(i), r_img_batch, r_ctg_batch, f_noise, f_ctg)

            progress_bar.set_description(
                f"Epoch {epoch+1:3d}/{EPOCH:3d} | " + 
                f"train_gen_loss: {train_gen_loss.result():10.7f}, " + 
                f"train_dis_loss: {train_dis_loss.result():10.7f} | " + 
                f"learning_rate: {optimizer._decayed_lr('float32').numpy():.15f}")

        gen_losses_history.append(train_gen_loss.result())
        dis_losses_history.append(train_dis_loss.result())
        lr_history.append(optimizer._decayed_lr('float32').numpy())

        plot_history(CKPT_DIR, {
            'gen_loss': gen_losses_history,
            'dis_loss': dis_losses_history,
            'lr'      : lr_history,
        })

        valid_img = generator(valid_noise, valid_ctg)
        plot_images(valid_img, valid_ctg, CKPT_DIR,
                    filename=f"valid-{epoch+1:03d}",
                    title=f"Validation of Epoch {epoch+1:03d}")

        if (epoch+1) % 10 == 0:
            plot_gif(CKPT_DIR)
            save_path = manager.save()
            checkpoint.restore(save_path=save_path).assert_consumed()
            # del save_path

        # del progress_bar, r_imgs, r_ctgs, r_img_batch, r_ctg_batch, noise, f_ctg, valid_img
        # gc.collect()

    return


''' Execution '''
if __name__ == '__main__':
    main()