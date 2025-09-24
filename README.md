# Understanding Knowledge Distillation by Transferring Symmetries

<b>Authors (alphabetical order):</b> Yves Bicker, Nicole Nobili, Patrick Odagiu, Fabian Schrag.

Knowledge distillation is used in an attempt to transfer the invariances exhibited by specific neural networks into simpler models that initially lack these structural symmetires, but could, in principle, learn them. It is observed that the simpler models fail at learning these invariances. To evaluate the scope of our finding and fortify our claims, we perform experiments on three distinct datasets and explore two different types of invariances: translational invariance on MNIST and ModelNet40 as well as permutation invariance on high-energy jet classification data.

## Installation & running

clone the repo
```bash
   git clone https://github.com/Nicole-Nobili/distillinginvariances.git
```

The repository consists of independent experiments conducted over three datasets organized into three folders: jetid_experiments, mnist_experiments and modelnet_experiments. Instructions on how to run each indivual experiment are provided in the README within the respective folder.


