import matplotlib.pyplot as plt
import tensorflow as tf
def plot_metrics(H, plotPath,metric=None):
    if(isinstance(H,tf.keras.callbacks.History)):
        H = H.history
	plt.style.use("ggplot")
	plt.figure()
    if metric is None:
        for k in H.keys():
            plt.plot(H[k], label=k)
        plt.ylabel("metrics")
    else:
        plt.plot(H[metric], label=metric)
        plt.ylabel(metric)
    plt.legend(loc="auto")
    plt.xlabel("Epoch #")
    plt.savefig(plotPath)
    plt.close()