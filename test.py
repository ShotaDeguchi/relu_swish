import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-5, 5, 256)
y_linear = tf.keras.activations.linear(.5 * x)
y_relu   = tf.keras.activations.relu(x)
y_swish1   = tf.nn.silu(features=x, beta=1/5)
y_swish2   = tf.nn.silu(features=x, beta=1/2.5)
y_swish3   = tf.nn.silu(features=x, beta=1.)
y_swish4   = tf.nn.silu(features=x, beta=2.5)
y_swish5   = tf.nn.silu(features=x, beta=5.)

y_swish_linear = tf.nn.silu(features=x, beta=1/9999.)
y_swish_relu   = tf.nn.silu(features=x, beta=9999.)

plt.figure(figsize=(8, 8))

#plt.plot(x, y_linear, alpha=.3, c="k", lw=3, label="linear")
#plt.plot(x, y_swish3, alpha=.7, linestyle="--", label="swish (beta=1.)")
#plt.plot(x, y_swish2, alpha=.7, linestyle="--", label="swish (beta=1/2.5)")
#plt.plot(x, y_swish1, alpha=.7, linestyle="--", label="swish (beta=1/5)")
#plt.plot(x, y_swish_linear, alpha=.7, c="r", linestyle="--", label="swish (beta -> 0.)")

plt.plot(x, y_relu,   alpha=.3, c="k", lw=5, label="relu")
plt.plot(x, y_swish3, alpha=.7, linestyle="--", label="swish (beta=1.)")
plt.plot(x, y_swish4, alpha=.7, linestyle="--", label="swish (beta=2.5)")
plt.plot(x, y_swish5, alpha=.7, linestyle="--", label="swish (beta=5.)")
plt.plot(x, y_swish_relu, alpha=.7, c="r", linestyle="--", label="swish (beta -> inf)")

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(alpha=.5)
plt.legend(loc="upper left")
plt.show()


