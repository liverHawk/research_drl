import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Discriminator(nn.Module):
    def __init__(self, input_size=784, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.use_sigmoid = True  # デフォルトはSigmoidを使用
    
    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the input
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        
        # ロス関数に応じてSigmoidの使用を制御
        if self.use_sigmoid:
            return nn.Sigmoid()(x)
        else:
            return x  # 生の出力値（Hinge Loss、WGAN用）
    

class Generator(nn.Module):
    def __init__(self, noise_size=128, hidden_size=256, output_size=784):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        if x.shape[1] == 784:
            x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)


class GAN:
    def __init__(
        self,
        noise_size=128,
        lr=0.0002,
        betas=(0.5, 0.999),
        loss_type='bce',
        save=True,
        d_input_size=784,
        d_hidden_size=256,
        g_hidden_size=256,
        g_output_size=784
    ):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.noise_size = noise_size
        self.loss_type = loss_type
        self.generator = Generator(
            noise_size=noise_size,
            hidden_size=g_hidden_size,
            output_size=g_output_size
        ).to(self.device)
        self.discriminator = Discriminator(
            input_size=d_input_size,
            hidden_size=d_hidden_size
        ).to(self.device)
        self.save = save
        
        # ロス関数の選択（WGAN-GPを追加）
        if loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None
        elif loss_type == 'wgan-gp':  # WGAN-GPを追加
            self.criterion = None
        elif loss_type == 'hinge':
            self.criterion = None
        
        # Hinge LossやWGANの場合はSigmoidを無効化
        if loss_type in ['hinge', 'wgan', 'wgan-gp']:
            self.discriminator.use_sigmoid = False

        # オプティマイザーの設定
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        if loss_type in ['wgan', 'wgan-gp']:
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.9))  # WGAN-GPではAdamを使用
        else:
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST(root='./test_data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    
    def wgan_loss(self, real_outputs, fake_outputs, fake_targets_for_gen=None):
        """Wasserstein GAN Loss"""
        if fake_targets_for_gen is not None:
            # ジェネレータのロス: 偽画像の出力を最大化
            return -torch.mean(fake_outputs)
        else:
            # 識別器のロス: 本物を最大化、偽物を最小化
            return torch.mean(fake_outputs) - torch.mean(real_outputs)
    
    def hinge_loss(self, real_outputs, fake_outputs, fake_targets_for_gen=None):
        """Hinge Loss for GAN (生の出力値を期待)"""
        if fake_targets_for_gen is not None:
            # ジェネレータのロス: 偽画像の出力を最大化
            return -torch.mean(fake_outputs)
        else:
            # 識別器のロス: max(0, 1-real) + max(0, 1+fake)
            real_loss = torch.mean(nn.ReLU()(1.0 - real_outputs))
            fake_loss = torch.mean(nn.ReLU()(1.0 + fake_outputs))
            return real_loss + fake_loss
    
    def gradient_penalty(self, real_samples, fake_samples):
        """WGAN-GPのグラディエントペナルティ"""
        batch_size = real_samples.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        # 実画像と偽画像の間の点をサンプリング
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        # 勾配を計算
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 勾配ペナルティを計算
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def discriminator_loss_calcurate(self, real_inputs, real_outputs, fake_inputs, fake_outputs):
        if self.loss_type == 'bce':
            real_label = torch.ones(real_inputs.shape[0], 1).to(self.device)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(self.device)
            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            D_loss = self.criterion(outputs, targets)
        
        elif self.loss_type == 'mse':
            real_label = torch.ones(real_inputs.shape[0], 1).to(self.device)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(self.device)
            real_loss = self.criterion(real_outputs, real_label)
            fake_loss = self.criterion(fake_outputs, fake_label)
            D_loss = real_loss + fake_loss
        
        elif self.loss_type == 'wgan':
            D_loss = self.wgan_loss(real_outputs, fake_outputs)
        
        elif self.loss_type == 'wgan-gp':
            # WGAN損失 + グラディエントペナルティ
            wgan_loss = self.wgan_loss(real_outputs, fake_outputs)
            gp = self.gradient_penalty(real_inputs, fake_inputs)
            lambda_gp = 20.0  # グラディエントペナルティの重み
            D_loss = wgan_loss + lambda_gp * gp
        
        elif self.loss_type == 'hinge':
            D_loss = self.hinge_loss(real_outputs, fake_outputs)
        return D_loss
    
    def generator_loss_calcurate(self, fake_inputs, fake_outputs):
        if self.loss_type == 'bce':
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(self.device)
            G_loss = self.criterion(fake_outputs, fake_targets)
        
        elif self.loss_type == 'mse':
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(self.device)
            G_loss = self.criterion(fake_outputs, fake_targets)
        
        elif self.loss_type in ['wgan', 'wgan-gp']:  # WGAN-GPも同じ生成器損失
            G_loss = self.wgan_loss(None, fake_outputs, fake_targets_for_gen=True)
        
        elif self.loss_type == 'hinge':
            G_loss = self.hinge_loss(None, fake_outputs, fake_targets_for_gen=True)
        return G_loss
    
    def _check_training_health(self, d_loss, g_loss):
        """トレーニングの健康状態をチェック"""
        if self.loss_type == 'bce':
            if d_loss < 0.1 and g_loss > 3.0:
                print("⚠️ 識別器が強すぎます。生成器の学習率を上げるか、識別器の学習頻度を下げてください。")
            elif d_loss > 2.0 and g_loss < 0.1:
                print("⚠️ ジェネレータが強すぎます。識別器の学習率を上げてください。")
            elif g_loss < 0.01:
                print("⚠️ 生成器損失が極めて小さいです。モード崩壊の可能性があります。")
        
        elif self.loss_type == 'mse':
            if d_loss < 0.1 and g_loss > 2.0:
                print("⚠️ MSE: 識別器が強すぎます。")
            elif g_loss < 0.01:
                print("⚠️ MSE: 生成器損失が極めて小さいです。")
        
        elif self.loss_type == 'wgan':
            if abs(d_loss) < 0.1:
                print("⚠️ WGAN損失が小さすぎます。クリッピング値を調整してください。")
            elif g_loss > -0.01:  # WGANでは負の値が良い
                print("⚠️ WGAN生成器の性能が悪いです。学習率やクリッピング値を調整してください。")
        
        elif self.loss_type == 'hinge':
            if d_loss < 0.05:
                print("⚠️ Hinge識別器損失が小さすぎます。")
            if g_loss > -0.01:  # マイナス値で判定
                print("⚠️ Hingeジェネレータの性能が悪いです。(G_loss: {:.3f})".format(g_loss))
            elif g_loss < -2.0:  # 極端に小さい場合
                print("⚠️ Hingeジェネレータ損失が極めて小さいです。モード崩壊の可能性があります。")
            elif g_loss < -0.8:  # 良い範囲
                print("✅ Hingeジェネレータが良好です。(G_loss: {:.3f})".format(g_loss))

    def train(self, epochs=50):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_d_losses = []
            epoch_g_losses = []
            
            for idx, batch in enumerate(self.dataloader):
                idx += 1

                # バッチが辞書形式かテンソル形式かを判定
                if isinstance(batch, dict):
                    # 辞書形式の場合（FlowDatasetの場合）
                    real_inputs = batch['data'].to(self.device)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    # タプル/リスト形式の場合（MNISTなどの場合）
                    imgs, _ = batch
                    real_inputs = imgs.to(self.device)
                else:
                    # 直接テンソルの場合
                    real_inputs = batch.to(self.device)

                # 入力を平坦化（必要に応じて）
                if real_inputs.dim() == 4:  # 画像データ（B, C, H, W）の場合
                    real_inputs = real_inputs.view(real_inputs.size(0), -1)
                elif real_inputs.dim() == 3:  # 3次元の場合
                    real_inputs = real_inputs.view(real_inputs.size(0), -1)

                # 識別器の学習（WGAN系では複数回実行）
                n_critic = 5 if self.loss_type in ['wgan', 'wgan-gp'] else 1
                
                for _ in range(n_critic):
                    real_outputs = self.discriminator(real_inputs)

                    noise = (torch.rand(real_inputs.shape[0], self.noise_size) - 0.5) / 0.5
                    noise = noise.to(self.device)
                    fake_inputs = self.generator(noise)
                    fake_outputs = self.discriminator(fake_inputs.detach())

                    D_loss = self.discriminator_loss_calcurate(real_inputs, real_outputs, fake_inputs, fake_outputs)

                    self.optimizer_D.zero_grad()
                    D_loss.backward()
                    self.optimizer_D.step()

                    # WGANの場合のみ重みクリッピング（WGAN-GPでは不要）
                    if self.loss_type == 'wgan':
                        for p in self.discriminator.parameters():
                            p.data.clamp_(-0.05, 0.05)

                # ジェネレータの学習
                noise = (torch.rand(real_inputs.shape[0], self.noise_size)-0.5)/0.5
                noise = noise.to(self.device)
                fake_inputs = self.generator(noise)
                fake_outputs = self.discriminator(fake_inputs)

                G_loss = self.generator_loss_calcurate(fake_inputs, fake_outputs)

                self.optimizer_G.zero_grad()
                G_loss.backward()
                self.optimizer_G.step()

                # ロス値を記録
                epoch_d_losses.append(D_loss.item())
                epoch_g_losses.append(G_loss.item())

                if idx % 100 == 0 or idx == len(self.dataloader):
                    if self.loss_type == 'hinge':
                        print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f} (負の値が良好)'.format(
                            epoch+1, idx, D_loss.item(), G_loss.item()))
                    elif self.loss_type == 'wgan-gp':
                        print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f} (WGAN-GP)'.format(
                            epoch+1, idx, D_loss.item(), G_loss.item()))
                    else:
                        print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(
                            epoch+1, idx, D_loss.item(), G_loss.item()))

            # エポック終了時の統計情報
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            print(f"Epoch {epoch+1} 平均損失 - D: {avg_d_loss:.3f}, G: {avg_g_loss:.3f}")
            
            self._check_training_health(avg_d_loss, avg_g_loss)

            if self.save and (epoch+1) % 10 == 0:
                torch.save(self.generator.state_dict(), f'Generator_{self.loss_type}_epoch_{epoch}.pth')
                print('モデルを保存しました。')
    
    def generate(self, num_samples=16, is_show=True):
        noise = (torch.rand(num_samples, self.noise_size) - 0.5) / 0.5
        noise = noise.to(self.device)

        with torch.no_grad():
            fake_data = self.generator(noise)
        
        print(fake_data)
        return
        # # ノイズから偽の画像を生成
        # noise = (torch.rand(num_samples, self.noise_size) - 0.5) / 0.5
        # noise = noise.to(self.device)
        
        # with torch.no_grad():
        #     fake_images = self.generator(noise)
        
        # # 生成された画像を表示
        # fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        # for i in range(num_samples):
        #     row = i // 4
        #     col = i % 4
        #     # 画像を正規化して表示用に変換 (-1, 1) -> (0, 1)
        #     img = (fake_images[i].cpu().detach().numpy().squeeze() + 1) / 2
        #     axes[row, col].imshow(img, cmap='gray')
        #     axes[row, col].axis('off')
        
        # plt.tight_layout()
        # if is_show:
        #     plt.show()
        
        #     # データローダーから実際のMNIST画像も表示したい場合
        #     print("実際のMNIST画像:")
        #     data_iter = iter(self.dataloader)
        #     real_images, _ = next(data_iter)
            
        #     fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        #     for i in range(min(16, real_images.shape[0])):
        #         row = i // 4
        #         col = i % 4
        #         # 画像を正規化して表示用に変換 (-1, 1) -> (0, 1)
        #         img = (real_images[i].numpy().squeeze() + 1) / 2
        #         axes[row, col].imshow(img, cmap='gray')
        #         axes[row, col].axis('off')
            
        #     plt.tight_layout()
        #     plt.show()
        # return plt