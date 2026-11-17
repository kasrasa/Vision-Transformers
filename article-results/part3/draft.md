# Vision Transformers, Part 3: Bringing Locality and Hierarchy Back to Attention

*Exploring how Swin Transformer organizes attention into local windows and builds hierarchical feature representations.*

In Part 2, I investigated whether better training could improve the performance of my custom Vision Transformer on Imagenette. Inspired by DeiT, I explored changes to the classifier, learning-rate scheduling, augmentation, and knowledge distillation from a CNN teacher.

The most substantial improvements came from augmentation and distillation. The original ViT from Part 1 achieved 57.68% accuracy on the held-out test set. With stronger augmentation and hard distillation, the ViT reached 75.90%, an improvement of 18.22 percentage points. The half-width ResNet-18 teacher achieved 79.21%, leaving a gap of 3.31 percentage points.

These experiments showed that training choices could substantially change the performance of the ViT. However, its underlying approach to processing images remained unchanged: it represented an image as a flat sequence of patch tokens and used global self-attention to exchange information between them.

This raises a different question: **What happens when the number of image tokens grows?**

With 16 × 16 patches, a 224 × 224 image produces a 14 × 14 grid containing 196 patch tokens. Increasing the resolution to 448 × 448 produces a 28 × 28 grid containing 784 tokens—four times as many.

Global attention calculates relationships between every pair of tokens. Excluding special tokens, the number of attention-score entries per head therefore increases from 196² = 38,416 to 784² = 614,656: a sixteenfold increase. This describes the growth of the attention matrix, rather than a measured sixteenfold increase in total runtime.

There is also a difference in how spatial relationships are built into the architecture. CNNs explicitly connect neighboring locations through convolutional kernels. Standard ViTs have fewer such spatial constraints: they can learn local relationships, but their attention layers do not enforce a local neighborhood.

[Swin Transformer](https://arxiv.org/abs/2103.14030) introduces a different organization. It computes attention within windows of neighboring patch tokens, shifts the window partition between successive blocks to enable communication across boundaries, and merges neighboring tokens between stages to construct a hierarchy of features. With fixed window sizes and embedding dimensions, attention computation scales linearly with the total number of tokens.

In this article, I will implement these mechanisms and investigate their effects separately. How much computation does window attention save? What does shifting contribute? And how does a hierarchical transformer compare with my existing ViT and half-width ResNet-18 under the same experimental conditions?

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Part 2 notebook and recorded held-out test results](../../part2.ipynb)
