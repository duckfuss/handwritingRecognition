# Intro

- Project start date: 25/05/24
- started during half term, idea was to create a neural network from scratch
- if possible reference my red notebook 2 for my notes - especially on the backprop derivatives
- did not use anyone else's code
- this is for future me before i forget!

# Instructions

- run the program called interface.py to get started
- interface2.py is unfinished as of 01/06/24
- the dataset can be changed from MNIST to FASHION MNIST by changing:
  - neuralNetwork.py: generateNetwork() → generateNetwork(dataType="FASHION")
  - interface.py: tdata.loadMNIST() → tdata.loadFashionMNIST()
- passing verbose=True into interface.py/recalc() will make it print the inputs, outputs and weights for each layer

## Keybindings

- these keybindings are for the interface

| Command                                          | KEY   |
| ------------------------------------------------ | ----- |
| clear drawing                                    | SPACE |
| insert next example from test data               | UP    |
| insert prev example from test data               | DOWN  |
| recalculate inputs (should happen automatically) | ENTER |
| draw                                             | LMB   |

# Resources used

- https://www.3blue1brown.com/lessons/backpropagation-calculus#computing-the-first-derivative (Main resource)
  - SUPER helpfull
- http://neuralnetworksanddeeplearning.com/ (but not the code sections)
- https://www.desmos.com/calculator/ccskgcoqgn

# Shortcomings

- Though the network scores very highly on the test data, when data is inputted by the user it often makes mistakes
  - this may be due to the fact the resulting inputs are subtly different, like the anti-aliasing has different fall-offs etc.
    - thus the mouse currently draws a 2 high line like a highlighter to more closely mimick the data
  - however IK for sure that it's a problem in the input (interface.py) program as when test data is loaded into the program it works 100%
  - interface2.py is about trying to more closely replicate the anti-aliasing of the training and testing data (MNIST)
- runs on CPU - gets quite hot!!!

# TODO

* [ ] pickle the training
* [ ] finish interface2 - which should emulate the MNIST data better for better mouse input results
* [ ] add picture/doodle recognition dataset
* [ ] try a convolutional network next
* [ ] implement into connect 4?? - long term goal
