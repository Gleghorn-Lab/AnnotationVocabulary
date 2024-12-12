# AnnotationVocabulary

This is the official repository of the paper [_Annotation Vocabulary (Might Be) All You Need_](https://doi.org/10.1101/2024.07.30.605924).

The data and models were developed using a proprietary internal developers package (udev), for which we've refactored a public version (udev_public). This repo is set up with the code intact, but currently, you will need to modify into your own files / file structure. For example, if you want to train a camp model you will have to place the udev_public file in a location that the train.py can see, or use `import sys` to make the parent folder available. We are planning on releasing some _ready to run_ scripts in the future.

## Models, training data
[Huggingface](https://huggingface.co/collections/GleghornLab/annotation-vocabulary-667c0e18efe26480c3ebe5f7)
## Downstream data
[Huggingface](https://huggingface.co/collections/GleghornLab/protein-fine-tuning-667dec58b0f62e5e07586bb2)

https://github.com/user-attachments/assets/482f8259-c7a8-481f-b871-dc90bfd8c52e

If you use any of our models, data, or code, please cite the following paper (and be aware of the GPL-3.0 license)
```
@article{hallee2024annotation,
      title={Annotation Vocabulary(Might Be) All You Need}, 
      author={Logan Hallee and Niko Rafailidis and Colin Horger and David Hong and Jason P. Gleghorn},
      year={2024},
      eprint={},
      archivePrefix={biorXiv},
      primaryClass={cs.LG}
}
```
