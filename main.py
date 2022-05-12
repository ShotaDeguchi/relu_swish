"""
********************************************************************************
main file to execute your program
********************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt

from config_gpu import config_gpu
from functions import *
from dnn import *

def main():
    # gpu configuration
    config_gpu(gpu_flg = 1)

    # problem setup
    p_id = 1
    xmin = -1.
    xmax =  1.
    nx   = 2 ** 8
    nx_  = 2 ** 4

    # params
    f_in   = 1
    f_out  = 1
    f_hid  = 5
    depth  = 3
    w_init = "He"
    b_init = "zeros"
    act    = "swish"
    beta   = 1.
    lr     = 5e-4
    opt    = "Adam"
    f_scl  = "minmax"
    laaf   = False
    d_type = "float32"
    d_type = tf.float32
    r_seed = 1234
    n_epc  = int(3e4)
    n_btc  = -1
    c_tol  = 1e-6
    es_pat = int(1e2)

    # prepare data
    x = np.linspace(xmin, xmax, nx)
    x_train = np.linspace(xmin, xmax, nx_).reshape(-1, 1)
    x_train = tf.convert_to_tensor(x_train, dtype=d_type)
    x_infer = np.linspace(xmin, xmax, nx).reshape(-1, 1)
    x_infer = tf.convert_to_tensor(x_infer, dtype=d_type)

    if p_id == 0:
        y = func0(x)
        y_train = func0_tf(x_train)
    elif p_id == 1:
        y = func1(x)
        y_train = func1_tf(x_train)
    elif p_id == 2:
        y = func2(x)
        y_train = func2_tf(x_train)
    else:
        raise NotImplementedError(">>>>> p_id")

    # compare
    if p_id == 0:
        raise NotImplementedError(">>>>> p_id")

    elif p_id == 1:
        # linear model for reference
        act = "linear"
        model_linear = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_linear.train(n_epc, n_btc, c_tol, es_pat)
        y_linear = model_linear.infer(x_infer)

        # swish
        act = "swish"
        beta = 1.
        lr = 5e-4 / beta
        lr = 5e-4 / (beta ** depth)
        model_swish1 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish1.train(n_epc, n_btc, c_tol, es_pat)
        y_swish1 = model_swish1.infer(x_infer)

        # swish
        act = "swish"
        beta = 1. / 2.5
        model_swish2 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish2.train(n_epc, n_btc, c_tol, es_pat)
        y_swish2 = model_swish2.infer(x_infer)

        # swish
        act = "swish"
        beta = 1. / 5.
        model_swish3 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish3.train(n_epc, n_btc, c_tol, es_pat)
        y_swish3 = model_swish3.infer(x_infer)

        # swish
        act = "swish"
        beta = 1. / 9999.
        model_swish4 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish4.train(n_epc, n_btc, c_tol, es_pat)
        y_swish4 = model_swish4.infer(x_infer)

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="c")
        plt.plot(x_infer, y_linear, label="linear", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.plot(x_infer, y_swish1, label="swish (beta=1.)",     alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish2, label="swish (beta=1./2.5)", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish3, label="swish (beta=1./5.)",  alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish4, label="swish (beta -> 0.)",  alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_linear.loss_log, label="linear", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.plot(model_swish1.loss_log, label="swish (beta=1.)",     alpha=.7, linestyle="--")
        plt.plot(model_swish2.loss_log, label="swish (beta=1./2.5)", alpha=.7, linestyle="--")
        plt.plot(model_swish3.loss_log, label="swish (beta=1./5.)",  alpha=.7, linestyle="--")
        plt.plot(model_swish4.loss_log, label="swish (beta -> 0.)",  alpha=.7, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

    elif p_id == 2:
        # relu model for reference
        act = "relu"
        model_relu = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_relu.train(n_epc, n_btc, c_tol, es_pat)
        y_relu = model_relu.infer(x_infer)

        # swish
        act = "swish"
        beta = 1.
        model_swish1 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish1.train(n_epc, n_btc, c_tol, es_pat)
        y_swish1 = model_swish1.infer(x_infer)

        # swish
        act = "swish"
        beta = 2.5
        model_swish2 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish2.train(n_epc, n_btc, c_tol, es_pat)
        y_swish2 = model_swish2.infer(x_infer)

        # swish
        act = "swish"
        beta = 5.
        model_swish3 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish3.train(n_epc, n_btc, c_tol, es_pat)
        y_swish3 = model_swish3.infer(x_infer)

        # swish
        act = "swish"
        beta = 9999.
        model_swish4 = DNN(
            x_train, y_train, 
            f_in, f_out, f_hid, depth, 
            w_init, b_init, act, beta, 
            lr, opt, f_scl, laaf, 
            d_type, r_seed
        )
        with tf.device("/device:GPU:0"):
            model_swish4.train(n_epc, n_btc, c_tol, es_pat)
        y_swish4 = model_swish4.infer(x_infer)

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label="function", alpha=.3, linestyle="-", lw = 5, c="c")
        plt.plot(x_infer, y_relu,   label="relu", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.plot(x_infer, y_swish1, label="swish (beta=1.)",  alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish2, label="swish (beta=2.5)", alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish3, label="swish (beta=5.)",  alpha=.7, linestyle="--")
        plt.plot(x_infer, y_swish4, label="swish (beta -> inf)", alpha=.7, linestyle="--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1.2, 1.2)
        plt.ylim(-2.5, 2.5)
        plt.grid(alpha=.5)
        plt.legend(loc="upper left")
        plt.savefig("./figures/approx_problem" + str(p_id) + ".png")

        plt.figure(figsize=(8, 4))
        plt.plot(model_relu  .loss_log, label="relu", alpha=.3, linestyle="-", lw = 5, c="k")
        plt.plot(model_swish1.loss_log, label="swish (beta=1.)",  alpha=.7, linestyle="--")
        plt.plot(model_swish2.loss_log, label="swish (beta=2.5)", alpha=.7, linestyle="--")
        plt.plot(model_swish3.loss_log, label="swish (beta=5.)",  alpha=.7, linestyle="--")
        plt.plot(model_swish4.loss_log, label="swish (beta -> inf)", alpha=.7, linestyle="--", c="r")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.grid(alpha=.5)
        plt.legend(loc="upper right")
        plt.savefig("./figures/loss_problem" + str(p_id) + ".png")

if __name__ == "__main__":
    main()

