# Char-RNN

RNN implementation taking inspiration from Andrew Karpathy's famous blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

The Char-RNN described in the post is trained a corpus of text (e.g. C++ code, Shakespeare) and after training produces text that is highly resembling of the training set.
A sequence of characters is input to the RNN, and after passing through the recurrent hidden layers, softmax is applied producing a probability distribution over all possible characters. The character with the highest probability is selected as the RNN's prediction and it's going to be added to the initial input sequence of characters for the next iteration through the network.

The Char-RNN is used for music composition is somewhat similar to the one described above, except that the input sequence is comprised of a sequence of pitches (e.g. A,B,G) and chords (e.g. CEG,ACE,DFA) and the RNN is going to predict the note, or chord using softmax.

Since exclusively pitches are used for prediction, the initial song is stripped off of note durations, and a standard note duration is assigned to each predicted note. For that reason, the RNN-RBM was implemented (see repo [here](https://github.com/stwykd/rnn-rbm)). This more powerful other architecture allows for mapping higher-dimensional probability distributions, allowing the network to predict note duration, among the other advantages.

In addition, the implementation uses LSTM-cells, which allowed the network to learn long-term musical structures, such as song forms, and produce melodic repetitions.

Please check out the compositions/ folder to hear the RNN's compositions.
