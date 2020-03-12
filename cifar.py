import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.gridspec as gridspec


def var_info(v):
    if not isinstance(v, str):
        print("The function var_info takes the name of the variable as a string!")
        sys.exit(0)
    print("{} is of type {} with shape {}".format(v, eval("type({})".format(v)), eval("{}.shape".format(v))))


def scoop_and_preprocess_data():
    global cifar_data, x_train, x_test, y_train, y_test, target_set
    global labels

    # labels scooped from https://github.com/zalandoresearch/fashion-mnist
    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    # grab fashion mnist data - in some custome class

    cifar_data = tf.keras.datasets.cifar10



    # pull out train and test in/out numpy arrays from class

    (x_train, y_train), (x_test, y_test) = cifar_data.load_data()

    # get feel for size and shape of things

    var_info("x_train")
    var_info("y_train")

    target_set = set(y_train.reshape(-1,))

    print("Num unique targets appears to be {}".format(len(target_set)))
    print(target_set)


def build_and_compile_model():
    global net_design
    # build network (model?) - a few cnn layers then normal layers

    inp_layer = tf.keras.layers.Input(shape=x_train[0].shape)
    net_layers = tf.keras.layers.Conv2D(29, (3, 3), strides=2, activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Conv2D(68, (3, 3), strides=2, activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Conv2D(108, (3, 3), strides=2, activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Flatten()(net_layers)
    net_layers = tf.keras.layers.Dropout(0.2)(net_layers)
    net_layers = tf.keras.layers.Dense(400, activation="relu")(net_layers)
    net_layers = tf.keras.layers.Dropout(0.2)(net_layers)
    net_layers = tf.keras.layers.Dense(len(target_set), activation="softmax")(net_layers)

    net_design = tf.keras.models.Model(inp_layer, net_layers)

    # compile: i.e. specify the learning mechanism - grad desc and error func sparce-cat-x-ent

    net_design.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


def do_the_learning():
    global training_history_data
    # learn! (fit)... specify data and epochs

    training_history_data = net_design.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=7)


def show_samples(xarr, yarr, netout):
    n = xarr.shape[0]
    rows = 1+int((n-1)/5)
    cols = 5

    fig = plt.figure()
    fig.suptitle("A collection of {} images".format(n))

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
        targ = np.zeros((10,))
        targ[y] = 1
        ax.bar(list(range(10)), targ, color="red")
        ax.bar(list(range(10)), nout, color="green")
        plt.xticks(list(range(10)), labels=list(range(10)))
        plt.yticks([])
        ax.set_ylim([0, 1])
        ax.set_xlim([-.5, 9.5])

    plt.show()


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


def disp_learn_prog(the_loss, the_val_loss, the_accuracy, the_val_accuracy):
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


scoop_and_preprocess_data()
show_n = 20
dummie_out = np.zeros((show_n, len(target_set)))

#show_an_input_output_pair(x_train[:show_n], y_train[:show_n], dummie_out)

#scratch = input("Learn from scratch?")
scratch = "n"

if scratch[:1] == "y":
    build_and_compile_model()
    do_the_learning()
    net_design.save('cifar.h5')
    save_learning_history()

    disp_learn_prog(training_history_data.history['loss'],
                    training_history_data.history['val_loss'],
                    training_history_data.history['accuracy'],
                    training_history_data.history['val_accuracy'])

else:
    net_design = tf.keras.models.load_model('cifar.h5')
    loss, val_loss, accuracy, val_accuracy = load_learning_history()
    disp_learn_prog(loss, val_loss, accuracy, val_accuracy)

predicted_answer_as_10_floats = net_design.predict(x_test)
show_samples(x_test[:show_n], y_test[:show_n], predicted_answer_as_10_floats)
