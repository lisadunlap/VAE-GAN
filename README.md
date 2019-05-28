# VAE-GAN
conditional VAE-GAN used on MNIST digits 

A VAE-GAN is a Variational Autoencoder combined with a Generative Adversarial Network

![Variational Autoencoder](https://i.ibb.co/Bsq0HjT/Screen-Shot-2019-05-28-at-10-44-45-AM.png)

![VAE-GAN](https://i.ibb.co/1m6YHr1/Screen-Shot-2019-05-28-at-10-43-26-AM.png)

We use a VAE-GAN on MNIST digits to create **counterfactual explanations**, which is explanations in relation to an alternate class label. 

For example, here we use this method to explain MNIST digit classifications with respect to the label 8. As seen below, we have the original images, the generated images, and the altered images generated such that the model now classifies them as an 8 instead of their original label. Alpha is a variable that controls the amount of augmentatiton (a higher alpha means more augmentation, so the numbers will look more and more like 8's). 

![Example of turning inputs to 8's](https://i.ibb.co/RD86g3B/Screen-Shot-2019-05-28-at-11-04-59-AM.png)

We hope that this explanation could be helpful in scenerios where the user is often looking for an explanation because they expected a different classififcation, and by seeing what would need to be changed to cause that classification, the user would understand why the model made its decision. 
