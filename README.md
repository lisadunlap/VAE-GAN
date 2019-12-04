# VAE-GAN
We present Conditional VAE-GAN used on MNIST digits to generate counterfactual examples.

A VAE-GAN is a Variational Autoencoder combined with a Generative Adversarial Network

![Variational Autoencoder](https://i.ibb.co/Bsq0HjT/Screen-Shot-2019-05-28-at-10-44-45-AM.png)

![VAE-GAN](https://i.ibb.co/1m6YHr1/Screen-Shot-2019-05-28-at-10-43-26-AM.png)

We use a VAE-GAN on MNIST digits to create **counterfactual explanations**, or explanations with respect to an alternate class label. For example, why did the network say this digit was a 3 instead of an 8?
This is done by altering the one-hot class vector so that it has nonzero values in both the predicted class index as well as the counterfactual class index. By doing this we are essentially forcing the generator to decode the encoding in the domains of both classes.  

For example, here we use this method to explain MNIST digit classifications with respect to the label 8. As seen below, we have the original images, the generated images, and the altered images generated such that the model now classifies them as an 8 instead of their original label. Alpha is a variable that controls the amount of augmentatiton (a higher alpha means more augmentation, so the numbers will look more and more like 8's). 

![Example of turning inputs to 8's](https://i.ibb.co/RD86g3B/Screen-Shot-2019-05-28-at-11-04-59-AM.png)

We hope that this explanation could be helpful in scenerios where the user is often looking for an explanation because they expected a different classififcation, and by seeing what would need to be changed to cause that classification, the user would understand why the model made its decision. 

Although results are promising on low dimension images like MNIST, VAE-GANs are not able to keep the resolution of more complex images, resulting in a fuzzy generated image. It is also unclear as to what a counterfactual explanation would look like in high dimension images (for example, how do we explain why the model predicted cat instead of dog?). It could be that for such situations, a generative approach is not useful. 

Our work was motivated by [this paper](https://papers.nips.cc/paper/7340-explanations-based-on-the-missing-towards-contrastive-explanations-with-pertinent-negatives.pdf), which also explored counterfactuals but through a black-box approach using data augmentation. 