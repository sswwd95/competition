import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

def get_tr_data(SEED):    
    tr_data = pd.read_csv("../dacon7/train.csv", index_col = 0).values
    tr_data = shuffle(tr_data, random_state = 77)

    tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
    # letter부터 끝까지 
    tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))
    # 0번째 :id
    return tr_X, tr_Y


def get_ts_data():
    ts_data = pd.read_csv("../dacon7/test.csv", index_col = 0).values
    ts_X = tf.convert_to_tensor(ts_data[:, 1:], dtype = tf.float32)
    # letter부터 끝까지 

    return ts_X

IMAGE_SIZE = [28, 28] # (784사이즈를 변환)
RESIZED_IMAGE_SIZE = [256, 256]

# 값을 구간 [0,1]로 맞추는 정규화 작업
@tf.function
def _reshape_and_resize_tr(flatten_tensor, label):
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor, label


@tf.function
def _reshape_and_resize_ts(flatten_tensor): # without label
    image_tensor = tf.reshape(flatten_tensor, (*IMAGE_SIZE, 1))
    image_tensor = tf.keras.layers.experimental.preprocessing.Resizing(*RESIZED_IMAGE_SIZE)(image_tensor)
    image_tensor = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(image_tensor)
    return image_tensor

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

def print_shapes(l, l_name, end = False):
    assert type(l) is list

    print(f"len({l_name}) = {len(l)}")
    for i in range(len(l)):
        print(f"{l_name}[{i}]: {l[i]}")
    if not end:
        print()
    

def make_dataset(X, Y, type):
    if type == "train":
        assert X.shape[0] == Y.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)
                            ).map(_reshape_and_resize_tr, num_parallel_calls = AUTO
                            ).cache(
                            ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)   

    elif type == "validation":
        assert X.shape[0] == Y.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)
                            ).map(_reshape_and_resize_tr, num_parallel_calls = AUTO
                            ).cache(
                            # ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)

    elif type == "test":
        assert Y is None
        dataset = tf.data.Dataset.from_tensor_slices((X)
                            ).map(_reshape_and_resize_ts, num_parallel_calls = AUTO
                            ).cache(
                            # ).shuffle(20000, reshuffle_each_iteration = True
                            ).batch(BATCH_SIZE
                            ).prefetch(AUTO)
                            
    else:
        raise ValueError(f"Unknown type: {type}")

    return dataset

def make_fold(X, Y, K = 5):
    assert X.shape[0] == Y.shape[0]
    assert len(list(Y.shape)) == 1

    fold_size = [X.shape[0] // K] * (K - 1) + [X.shape[0] - (X.shape[0] // K) * (K - 1)]

    splited_X = tf.split(X, fold_size, 0) # list
    splited_Y = tf.split(Y, fold_size, 0) # list

    tr_list = [(tf.concat(splited_X[:i] + splited_X[i+1:], axis = 0), tf.concat(splited_Y[:i] + splited_Y[i+1:], axis = 0)) for i in range(K)]
    vl_list = [(splited_X[i], splited_Y[i]) for i in range(K)]

    tr_datasets = [make_dataset(tr_X, tr_Y, "train") for tr_X, tr_Y in tr_list]
    vl_datasets = [make_dataset(vl_X, vl_Y, "validation") for vl_X, vl_Y in vl_list]

    return tr_datasets, vl_datasets

tr_data = pd.read_csv("../dacon7/train.csv", index_col = 0).values
# tr_data = shuffle(tr_data, random_state = SEED)

tr_X = tf.convert_to_tensor(tr_data[:, 2:], dtype = tf.float32)
tr_Y = tf.squeeze(tf.convert_to_tensor(tr_data[:, 0], dtype = tf.int32))

aa = tf.reshape(tr_X[318], (28, 28, 1))
result1 = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(aa)
result2 = tf.keras.layers.experimental.preprocessing.Resizing(128, 128)(aa)
result3 = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(aa)

answer = tr_Y[318]


from matplotlib import pyplot as plt

plt.figure(figsize = (12, 4), dpi = 80)

for i, img in enumerate([aa, result1, result2, result3]):
    plt.subplot(1, 4, i + 1)
    plt.imshow(tf.squeeze(img), cmap = "jet")
    plt.title(f"{answer} / {tf.squeeze(img).shape}")
    
plt.tight_layout()
plt.show()

aa = tf.reshape(tr_X, (-1, 28, 28, 1))
result1 = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(aa)
result2 = tf.keras.layers.experimental.preprocessing.Resizing(128, 128)(aa)
result3 = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(aa)

plt.figure(figsize = (12, 4), dpi = 80)

for i, img in enumerate([aa, result1, result2, result3]):
    
    plt.subplot(1, 4, i + 1)
    plt.imshow(tf.squeeze(tf.math.reduce_sum(img, axis = 0)), cmap = "jet")
    # plt.title(f"{answer} / {tf.squeeze(img).shape}")
    
plt.tight_layout()
plt.show()

ts_data = pd.read_csv("./data/test.csv", index_col = 0).values
ts_X = tf.convert_to_tensor(ts_data[:, 1:], dtype = tf.float32)

aa = tf.reshape(ts_X, (-1, 28, 28, 1))
result1 = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(aa)
result2 = tf.keras.layers.experimental.preprocessing.Resizing(128, 128)(aa)
result3 = tf.keras.layers.experimental.preprocessing.Resizing(256, 256)(aa)

# answer = tr_Y

plt.figure(figsize = (12, 4), dpi = 80)

for i, img in enumerate([aa, result1, result2, result3]):
    
    plt.subplot(1, 4, i + 1)
    plt.imshow(tf.squeeze(tf.math.reduce_sum(img, axis = 0)), cmap = "jet")
    # plt.title(f"{answer} / {tf.squeeze(img).shape}")
    
plt.tight_layout()
plt.show()

def bn_ReLU_conv2D(x, filters, kernel_size, 
                   strides = 1, padding = "same", weight_decay = 1e-4):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters = filters, 
        kernel_size = kernel_size, 
        strides = strides,
        padding = padding,
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(x)
    
    return x


def transition_block(x):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(x.shape[-1] // 2, 1, padding = "same")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), strides = 2)(x)

    return x


def dense_block(x, num_conv, growth_rate):
    for i in range(num_conv):
        residual = x
        x = bn_ReLU_conv2D(x, 4 * growth_rate, 1)
        x = bn_ReLU_conv2D(x, growth_rate, 3)
        x = tf.keras.layers.Concatenate(axis = -1)([x, residual])

    return x

def create_NN(model_name, dropout_rate = 0.3, growth_rate = 32):
    model_input = tf.keras.layers.Input((*RESIZED_IMAGE_SIZE, 1))

    ## Pre-processing
    x = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0.1), (-0.2, 0.1), fill_mode = "constant")(model_input)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode = "constant")(x)

    ## Entry Flow
    x = tf.keras.layers.Conv2D(2 * growth_rate, 7, strides = 2, padding = "same")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides = 2, padding = "same")(x)


    ## Middle Flow
    aux_outputs = []
    for i, num_conv in enumerate([6, 12, 24, 16]):
        x = dense_block(x, num_conv, growth_rate)

        if i is not 3: 
            x = transition_block(x)
            x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)

        if i == 2:
            y = tf.keras.layers.GlobalAveragePooling2D()(x)
            y = tf.keras.layers.Dense(y.shape[-1])(y)
            y = tf.keras.layers.Lambda(lambda y: tf.math.l2_normalize(y, axis = 1), name = "aux")(y) # L2 normalize embeddings
            aux_outputs.append(y)


    ## Exit Flow
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(x.shape[-1])(x)
    real_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis = 1), name = "real")(x) # L2 normalize embeddings

    model = tf.keras.Model(
        inputs = model_input,
        outputs = aux_outputs + [real_output],
        name = model_name)
    
    return model

tmp_model = create_NN("tmp_model")
tmp_model.summary()

tf.keras.utils.plot_model(tmp_model, show_shapes = True, show_layer_names = False)

del tmp_model

def get_opt(init_lr = 3e-3):
    radam = tfa.optimizers.RectifiedAdam(
        lr = init_lr, warmup_proportion = 0, min_lr = 1e-5, weight_decay = 1e-4)
    ranger = tfa.optimizers.Lookahead(radam)

    return ranger

# Define callbacks
def scheduler(epoch, lr):
    # lr = 0.96 * lr for every 4 epochs
    if not epoch:
        return lr
    elif not (epoch % 4):
        return 0.96 * lr
    else:
        return lr


def get_callbacks(GCS_PATH, model_name):
    # reduce learning rate callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # checkpoint callback
    checkpoint_path = os.path.join(
        GCS_PATH, "ckpt", NOTEBOOKNAME, model_name, "cp-{epoch:03d}-{val_real_loss:.4f}.ckpt")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose = 2, monitor = "val_real_loss",
        save_weights_only = True, save_best_only = True)

    return [lr_callback, cp_callback]