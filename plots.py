import os
import matplotlib.pyplot as plt

def plot(list_name, plot_name, model_name):
    plt.figure()
    plt.plot(list_name, label=f'{model_name}_{plot_name}')
    plt.gca().set_xlabel('Epoch')
    plt.gca().set_ylabel(f'{plot_name}')
    plt.gca().set_title(f'{model_name}_{plot_name} Curve')
    plt.legend()

    #if plots folder doesn't exist, then
    if not os.path.exists('plots'):
      os.makedirs('plots')

    plt.savefig(f'plots/curve_{model_name}_{plot_name}')
    plt.show()
