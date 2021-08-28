# AbstractArtGAN
There is my summer 2021 project about different GAN approaches in generating abstract paintings.
Архитектура дискриминатора и генератора, которую я использовал в каждом варианте гана (в LSGAN и WCGAN у дискриминатора отсутствовала сигмоидная функция активации на выходе)

Discriminator:
Sequential(
  (0): Conv2d(3, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (1): LeakyReLU(negative_slope=0.2, inplace=True)
  (2): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (3): LeakyReLU(negative_slope=0.2, inplace=True)
  (4): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (5): LeakyReLU(negative_slope=0.2, inplace=True)
  (6): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  (7): Flatten(start_dim=1, end_dim=-1)
  (8): Sigmoid()
)

Generator:
Sequential(
  (0): ConvTranspose2d(128, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): LeakyReLU(negative_slope=0.2)
  (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): LeakyReLU(negative_slope=0.2)
  (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (8): LeakyReLU(negative_slope=0.2)
  (9): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (10): Tanh()
)

DCGAN Loss:
Discriminator: binary_cross_entropy(D(G(z)), x)
Generator: binary_cross_entropy(D(G(z)), x)

LSGAN Loss:
Discriminator: 0.5 * (torch.mean((D(x) - 1) ** 2) + torch.mean(D(G(z)) ** 2))
Generator: 0.5 * torch.mean((D(G(z)) - 1)\**2)

WCGAN Loss: 
Discriminator: -(torch.mean(D(x)) - torch.mean(D(G(z))))
Generator: -torch.mean(D(G(z)))

Результаты:
DCGAN 
![DCGAN3500](https://user-images.githubusercontent.com/68852747/131214016-4cd73527-415d-4a6e-a9e0-2ca9265de5b3.png)
LSGAN
![LSGAN](https://user-images.githubusercontent.com/68852747/131214019-3ca5b23e-0401-47a2-b4e3-3f93f1301196.png)
WCGAN
![WCGAN3500](https://user-images.githubusercontent.com/68852747/131214020-25879564-c58b-4a4a-9564-06b74258e950.png)
