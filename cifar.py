import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import time


def var_info(v):
    if not isinstance(v, str):
        print("The function var_info takes the name of the variable as a string!")
        sys.exit(0)
    print("{} is of type {} with shape {}".format(v, eval("type({})".format(v)), eval("{}.shape".format(v))))


def prepare_for_fewer_categories():
    global labels
    global sub_selection
    global num_catagories_in_use

    # labels scooped from https://github.com/zalandoresearch/fashion-mnist
    org_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sub_selection = [3, 5]
    num_catagories_in_use = len(sub_selection)
    labels = []
    for i in sub_selection:
        labels.append(org_labels[i])
    print("Now only training net to recognise the following items: ", labels)


def convert_data_to_fewer_categories(x_trainZ, y_trainZ):
    global num_catagories_in_use
    x_t = np.array([])
    y_t = np.array([])

    hitctr = 0
    for i, (xelem, yelem) in enumerate(zip(x_trainZ, y_trainZ)):
        if i > 0 and (i % 5000) == 0:
            print(i)
        if yelem[0] in sub_selection:
            if hitctr == 0:

                y_t = yelem.copy()
                x_t = xelem.copy()

                x_t = np.reshape(x_t, [1, 32, 32, 3])
                y_t = np.reshape(y_t, [1, 1])
                print("!")
            else:
                x_t = np.vstack([x_t, np.reshape(xelem, [1, 32, 32, 3])])
                y_t = np.vstack([y_t, np.reshape(yelem, [1, 1])])

            hitctr += 1

    for elem in y_t:
        for i, e in enumerate(sub_selection):
            if elem[0] == e:
                elem[0] = i
                break

    return x_t, y_t


def scoop_and_preprocess_data():
    global cifar_data, x_train, x_test, y_train, y_test
    global labels
    global x_t, y_t
    global num_catagories_in_use

    # grab fashion mnist data - in some custome class

    cifar_data = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar_data.load_data()

    # shrink_training_data(100)

    prepare_for_fewer_categories()
    x_train, y_train = convert_data_to_fewer_categories(x_train, y_train)
    x_test, y_test = convert_data_to_fewer_categories(x_test, y_test)


def build_and_compile_model(size_of_first_dense_layer_param):
    global model
    # build network (model?) - a few cnn layers then normal layers

    INPUT_LAYER = tf.keras.layers.Input(shape=x_train[0].shape)
    with tf.name_scope("Convolutional layers"):
        net_layers = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", name="C32_A")(INPUT_LAYER)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", name="C_32_B")(net_layers)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.MaxPooling2D((2,2))(net_layers)

        net_layers = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu", name="C_64_A")(net_layers)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu", name="C_64_B")(net_layers)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.MaxPooling2D((2,2))(net_layers)

        net_layers = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu", name="C_128_A")(net_layers)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu", name="C_128_B")(net_layers)
        net_layers = tf.keras.layers.BatchNormalization()(net_layers)
        net_layers = tf.keras.layers.MaxPooling2D((2,2))(net_layers)

        net_layers = tf.keras.layers.Flatten()(net_layers)
        net_layers = tf.keras.layers.Dropout(0.2)(net_layers)
    with tf.name_scope("Dense layers"):
        net_layers = tf.keras.layers.Dense(size_of_first_dense_layer_param, activation="relu", name="dense_1000")(net_layers)
        net_layers = tf.keras.layers.Dropout(0.2)(net_layers)
        net_layers = tf.keras.layers.Dense(num_catagories_in_use, activation="softmax", name="second_dense_ie_the_output")(net_layers)

    model = tf.keras.models.Model(INPUT_LAYER, net_layers)

    # compile: i.e. specify the learning mechanism - grad desc and error func sparce-cat-x-ent

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    '''
    # https://keras.io/utils/#plot_model
    # seems to need graphviz installed (not a python library)
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96)
    '''


def do_the_learning(ep, logdirname):
    global training_history_data

    # log_dir = "logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = logdirname

    print("log_dir is [{}]".format(log_dir))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # learn! (fit)... specify data and epochs

    training_history_data = model.fit(x_train,
                                      y_train,
                                      validation_data=(x_test, y_test),
                                      epochs=ep,
                                      callbacks=[tensorboard_callback])


def show_samples(tit, xarr, yarr, netout):
    n = xarr.shape[0]
    rows = 1+int((n-1)/5)
    cols = 5

    fig = plt.figure()
    fig.suptitle("A collection of {} '{}' images".format(n, tit))

    gridspec_array = fig.add_gridspec(rows*3, cols)

    for i, (x, y, nout) in enumerate(zip(xarr, yarr, netout)):
        r = int(i/5) * 3
        c = i % 5
        ax = fig.add_subplot(gridspec_array[r:r+2, c])
        plt.xlabel("{}: {}".format(y[0], labels[y[0]]))
        plt.xticks([])
        plt.yticks([])
        ax.imshow(x)

        ax = fig.add_subplot(gridspec_array[r+2, c])
        targ = np.zeros((num_catagories_in_use,))
        targ[y] = 1
        ax.bar(list(range(num_catagories_in_use)), targ, color="red")
        ax.bar(list(range(num_catagories_in_use)), nout, color="green")
        plt.xticks(list(range(num_catagories_in_use)), labels=list(range(num_catagories_in_use)))
        plt.yticks([])
        ax.set_ylim([0, 1])
        ax.set_xlim([-.5, 9.5])

    plt.show()
    #plt.show(block=False)
    #plt.ion()
    #plt.pause(0.1)
    #plt.ioff()


def save_learning_history():
    with open('learnhist.txt', 'w') as f:
        f.write("{}\n".format(len(training_history_data.history['loss'])))
        for elem in training_history_data.history['loss']:
            f.write("{}\n".format(elem))
        for elem in training_history_data.history['val_loss']:
            f.write("{}\n".format(elem))
        for elem in training_history_data.history['accuracy']:
            f.write("{}\n".format(elem))
        for elem in training_history_data.history['val_accuracy']:
            f.write("{}\n".format(elem))


def load_learning_history():
    loss, val_loss, accuracy, val_accuracy = [], [], [], []
    with open('learnhist.txt', 'r') as f:
        line = f.readline(9999)
        elems = int(line)
        for elem in range(elems):
            line = f.readline(9999)
            loss.append(float(line))
        for elem in range(elems):
            line = f.readline(9999)
            val_loss.append(float(line))
        for elem in range(elems):
            line = f.readline(9999)
            accuracy.append(float(line))
        for elem in range(elems):
            line = f.readline(9999)
            val_accuracy.append(float(line))
    return loss, val_loss, accuracy, val_accuracy


def display_learning_progression(the_loss, the_val_loss, the_accuracy, the_val_accuracy):
    plt.suptitle("Learning history")
    plt.subplot(1, 2, 1)
    plt.title("loss measure...\n...the thing we try and minismise during learning")
    plt.plot(the_loss, label="loss (Sparse Cat X Ent)")
    plt.plot(the_val_loss, label='validation loss (Sparse Cat X Ent)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy... the fraction we get correct")
    plt.plot(the_accuracy, label='accuracy')
    plt.plot(the_val_accuracy, label='validation accuracy')

    plt.legend()
    plt.show(block = False)


def show_some_sample_images_without_network(n):
    dummie_out = np.zeros((n, num_catagories_in_use))
    show_samples("Training data",x_train[:n], y_train[:n], dummie_out)
    show_samples("Testing data", x_test[:n], y_test[:n], dummie_out)


def shrink_training_data(n):
    global x_train, y_train, shrunk
    shrunk = True
    x_train = x_train[:n]
    y_train = y_train[:n]


def print_any_warnings():
    global shrunk
    if shrunk:
        print("SHRINKING TRAINING DATA FOR QUICK TEST!!!!")
        print("SHRINKING TRAINING DATA FOR QUICK TEST!!!!")
        print("SHRINKING TRAINING DATA FOR QUICK TEST!!!!")
        print("SHRINKING TRAINING DATA FOR QUICK TEST!!!!")


def save_data_already_processed():
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)


def load_data_already_processed():
    global x_train, x_test, y_train, y_test
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")


def hyperstring(ep, size_of_first_dense_layer_param):
    ans = "logs"
    ans += "_ep{:02d}".format(ep)
    ans += "_fd{:04d}".format(size_of_first_dense_layer_param)
    return ans


# get the training data
shrunk = False
if False:
    scoop_and_preprocess_data()
    save_data_already_processed()
    print_any_warnings()
    print("Data loaded, preprocess and saved")
else:
    prepare_for_fewer_categories()
    load_data_already_processed()
    print("Preprocessed data loaded")

show_n = 50
# show_some_sample_images_without_network(show_n)

learn_from_scratch = True  # input("Learn from scratch? ")[:1].lower() =='y'


if learn_from_scratch:

    size_of_first_dense_layer_list = [50, 100 , 150, 200]

    doing_hyperperameter_truning = True
    epochs = 12
    for size_of_first_dense_layer in size_of_first_dense_layer_list:
        build_and_compile_model(size_of_first_dense_layer)
        logname = hyperstring(epochs, size_of_first_dense_layer)
        do_the_learning(epochs, logname)
        model.save('cifar{}.h5'.format(logname))
        if not doing_hyperperameter_truning:
            save_learning_history()
            display_learning_progression(training_history_data.history['loss'],
                                         training_history_data.history['val_loss'],
                                         training_history_data.history['accuracy'],
                                             training_history_data.history['val_accuracy'])
else:
    model = tf.keras.models.load_model('cifar.h5')
    loss, val_loss, accuracy, val_accuracy = load_learning_history()
    display_learning_progression(loss, val_loss, accuracy, val_accuracy)


predicted_answer_as_10_floats = model.predict(x_test)
show_samples("After training process",x_test[:show_n], y_test[:show_n], predicted_answer_as_10_floats)
input("Now PAK to end the program")