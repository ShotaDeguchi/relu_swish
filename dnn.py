"""
********************************************************************************
deep neural network
********************************************************************************
"""

import os
import time
import numpy as np
import tensorflow as tf

class DNN:
    def __init__(
        self, 
        x, y, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, beta, 
        lr = 5e-4, opt = "Adam", f_scl = "minmax", laaf = False, 
        d_type = tf.float32, r_seed = 1234
    ):
        # initialization
        super().__init__()
        self.f_in   = f_in     # input features
        self.f_out  = f_out    # output features
        self.f_hid  = f_hid    # hidden layer dim
        self.depth  = depth    # depth of dnn
        self.w_init = w_init   # weight initializer
        self.b_init = b_init   # bias initializer
        self.act    = act      # activation
        self.beta   = beta     # scaling factor for swish
        self.lr     = lr       # learning rate
        self.opt    = opt      # optimizer ("SGD" / "RMSprop" / "Adam")
        self.f_scl  = f_scl    # feature scaling ("linear" / "minmax" / "mean")
        self.laaf   = laaf     # locally adaptive activation function
        self.d_type = d_type   # data type
        self.r_seed = r_seed   # random seed

        print("\n************************************************************")
        print("********************     DELLO WORLD     *******************")
        print("************************************************************")

        # set a random seed
        self.random_seed(self.r_seed)

        # input - output pair
        self.x = x
        self.y = y

        # for feature scaling
        X = tf.concat([x], axis = 1)
        self.lb = tf.cast(tf.reduce_min (X, axis=0), dtype=self.d_type)
        self.ub = tf.cast(tf.reduce_max (X, axis=0), dtype=self.d_type)
        self.mn = tf.cast(tf.reduce_mean(X, axis=0), dtype=self.d_type)

        # build a deep neural network
        self.structure = [self.f_in] + (self.depth-1) * [self.f_hid] + [self.f_out]
        self.weights, self.biases, self.alphas, self.params = self.dnn_init(self.structure)
        self.optimizer = self.opt_alg(self.lr, self.opt)
        self.loss_log = []

    def random_seed(
        self, r_seed
    ):
        print(">>>>> random_seed")
        print("         random seed:", r_seed)
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        tf.random.set_seed(r_seed)

    def dnn_init(self, strc):
        print(">>>>> dnn_init")
        print("         f_in :", self.f_in)
        print("         f_out:", self.f_out)
        print("         f_hid:", self.f_hid)
        print("         depth:", self.depth)
        print(">>>>> weight_init")
        print("         initializer:", self.w_init)
        print(">>>>> bias_init")
        print("         initializer:", self.b_init)
        print(">>>>> act_func")
        print("         activation:", self.act)
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for d in range(0, self.depth):   # depth = self.depth
            w = self.weight_init(shape = [strc[d], strc[d + 1]], depth = d)
            b = self.bias_init  (shape = [      1, strc[d + 1]], depth = d)
            weights.append(w)
            biases .append(b)
            params .append(w)
            params .append(b)
            if self.laaf == True and d < self.depth - 1:
                a = tf.Variable(1., dtype = self.d_type, name = "a" + str(d))
                params.append(a)
            else:
                a = tf.constant(1., dtype = self.d_type)
            alphas .append(a)
        return weights, biases, alphas, params

    def weight_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_init")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.d_type), \
            dtype = self.d_type, name = "w" + str(depth)
            )
        return weight

    def bias_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias

    def opt_alg(
        self, lr, opt
    ):
        print(">>>>> opt_alg")
        print("         learning rate:", lr)
        print("         optimizer    :", opt)
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False)
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        else:
            raise NotImplementedError(">>>>> opt_alg")
        return optimizer

    def forward_pass(
        self, x
    ):
        # feature scaling
        if self.f_scl == "linear":
            z = x
        elif self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mn) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")
        # forward pass
        for d in range(0, self.depth - 1):
            w = self.weights[d]
            b = self.biases [d]
            a = self.alphas [d]

            # print("self.lb", self.lb)
            # print("self.ub", self.ub)

            # print("x:", x)
            # print("z:", z)
            # print("w:", w)

            u = tf.add(tf.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "linear":
                z = u
            elif self.act == "relu":
                z = tf.nn.relu(u)
            elif self.act == "tanh":
                z = tf.tanh(u)
            elif self.act == "swish":
                # z = tf.nn.silu(features=u, beta=self.beta)
                # z = tf.multiply(u, tf.sigmoid(u))
                z = tf.multiply(u, tf.sigmoid(self.beta * u))
            elif self.act == "gelu":
                z = tf.multiply(u, tf.sigmoid(1.702 * u))
            elif self.act == "mish":
                z = tf.multiply(u, tf.tanh(tf.nn.softplus(u)))
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")
        w = self.weights[-1]
        b = self.biases [-1]
        a = self.alphas [-1]
        u = tf.add(tf.matmul(z, w), b)
        u = tf.multiply(a, u)
        z = u   # identity mapping
        y = z
        return y

    @tf.function
    def loss_func(
        self, x, y
    ):
        y_ = self.forward_pass(x)
        loss = tf.reduce_mean(tf.square(y - y_))
        if self.laaf == True:
            loss += 1. / tf.reduce_mean(tf.exp(self.alphas))
        else:
            pass
        return loss

    @tf.function
    def loss_grad(
        self, x, y
    ):
        with tf.GradientTape(persistent=True) as tp:
            loss = self.loss_func(x, y)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad

    @tf.function
    def grad_desc(
        self, x, y
    ):
        loss, grad = self.loss_grad(x, y)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss

    def train(
        self, n_epc, n_btc, c_tol, es_pat
    ):
        print(">>>>> train")
        print("         n_epoch:", n_epc)
        print("         n_batch:", n_btc)
        print("         c_tlrnc:", c_tol)
        print("         es_pat :", es_pat)

        wait = 0
        loss_best = 9999
        t0 = time.time()
        if n_btc == -1:
            print(">>>>> executing full-batch training")
            for epc in range(n_epc):
                loss_epc = 0.
                loss_epc = self.grad_desc(self.x, self.y)
                self.loss_log.append(loss_epc)

                # monitor 
                if epc % 100 == 0:
                    elps = time.time() - t0
                    print("epc: %d, loss: %.6e, elps: %.3f"
                        % (epc, loss_epc, elps))
                    t0 = time.time()

                # # save 
                # if epc % 100 == 0:
                #     self.save(self.save_path + "model" + str(epc))

                # early stopping
                if loss_epc < loss_best:
                    loss_best = loss_epc
                    wait = 0
                else:
                    if wait >= es_pat:
                        print(">>>>> early stopping")
                        break
                    wait += 1

                # terminate if converged
                if loss_epc < c_tol:
                    print(">>>>> converging to the tolerance")
                    break

        else:
            print(">>>>> executing mini-batch training")
            raise NotImplementedError(">>>>> train")

    def infer(
        self, x
    ):
        print(">>>>> infer")
        y_ = self.forward_pass(x)
        return y_

