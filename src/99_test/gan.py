import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.gan import GAN
# import matplotlib.pyplot as plt

def all_types():
    loss_types = ['mse', 'wgan', 'hinge', 'bce']
    for loss in loss_types:
        print(f"Testing GAN with {loss} loss...")
        gan = GAN(lr=5e-4, loss_type=loss, save=False)
        print("Starting training...")
        finish = gan.train(epochs=40)
        print("Generating samples...")
        plt = gan.generate(is_show=False)
        if finish:
            plt.savefig(f"gan_training_{loss}_{finish}.png")
        else:
            plt.savefig(f"gan_samples_{loss}.png")
        plt.close()

def test_one():
    LOSS_TYPE = 'wgan-gp'
    print(f"Testing GAN with {LOSS_TYPE} loss...")
    gan = GAN(lr=1e-3, loss_type=LOSS_TYPE, save=False)
    steps = 0
    for i in range(5):
        finish = gan.train(epochs=10)
        plt = gan.generate(is_show=False)
        if finish:
            steps += finish
            plt.savefig(f"test/gan_training_{LOSS_TYPE}_{steps}.png")
        else:
            steps += 40
            plt.savefig(f"test/gan_samples_{LOSS_TYPE}_{steps}.png")
        plt.close()

def main():
    # all_types()
    test_one()


if __name__ == "__main__":
    main()