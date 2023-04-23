# ml-mini FFF (framework for fun)

This WIP has multiple humble goals:

- Deeper understanding of ML algorithms and structure. Lots of great ML frameworks for sure hidding the painful complexity. I need to understanding the painful complexity to become a better ML dev.
- Level up Math skills.
- Level up Rust programming knowledge.
- Commit into a future where I would use ML and/or Rust as a professional.
- Have fun classifying cat pictures

Steps:

- (IN PROGRESS) Forward and backpropagation framework. API as simple as possible considering the language used. Therefore, lots of macros
- (TO DO) Write convenient input and output nodes.
- (TO DO) Very simple binary image classifier using a single Logistic Regression node in hidden layers.
- (TO DO) Locally storing weights and biases.
- (TO DO) LeNet-5 (implenting other kind of nodes: convolution, pooling, dense etc...).
- (TO DO) Other kind of famous models.
- (TO DO) Letzgongue!

I'm not sure what I'm doing at this point = contains no test 💀

## API functionalities

### Design

Made simple using macros:

```rust
model!(
    input_layer!(convert_img_to_vector("/path/to/file")),
    dense!(
        layer!(
            10,
            logistic_regression,
        ),
        layer!(
            12,
            logistic_regression,
        )
    ),
    output_layer!(sigmoid)
)
.run()
.train(0.5, y_vector)
```

### Automatic matrix manipulation

Framework automatically compute size of weights and biases with respect to previous layer's size.

### Activation functions, nodes and classifiers

Stack of provided (TO DO) activation functions, nodes and classifiers, ready to use. Those can also be specified through closures (TO DO).
