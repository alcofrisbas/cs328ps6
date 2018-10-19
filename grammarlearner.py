import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

############# Loading function #############

def load_data(filename):
    '''
    Loads data from filename, assuming one sequence per line.
    Returns a list where each item is a string in the dataset.
    '''
    strings = None
    with open(filename, 'r') as fid:
        lines = fid.readlines()
        strings = [l.strip() for l in lines]
    return strings


############# Functions for converting between characters, indices, and one-hot tensors #############

def letter_to_index(letter, all_letters):
    '''
    Returns the index of the given letter in the string
    all_letters; fails if letter is not in all_letters.
    all_letters is a string containing all possible symbols in the grammar; each
    letter must only occur once in this string.
    '''
    return all_letters.find(letter)


def one_hot_to_index(tensor):
    '''
    Takes a "one-hot" vector (one spot is a 1 and all others are 0s)
    and returns a tensor representing the index that has the one.
    E.g., if a tensor represented [0., 0., 1., 0., 0., 0., 0.],
    this would return a tensor representing [2].
    Think of this as the inverse of one letter of the conversion
    that happens in seq_to_tensor.
    '''
    idx = np.nonzero(tensor.data.numpy().flatten())[0]
    assert len(idx) == 1
    return torch.LongTensor(idx)


def seq_to_tensor(sequence, all_letters):
    '''
    Turn a sequence of symbols into a tensor of shape <seq_length x 1 x n_letters>,
    where each symbol is a "one-hot" vector: one spot is a 1 and all others are 0s.
    all_letters is a string containing all possible symbols in the grammar; each
    letter must only occur once in this string.
    '''
    tensor = torch.zeros(len(sequence), 1, len(all_letters))
    for li, letter in enumerate(sequence):
        tensor[li][0][letter_to_index(letter, all_letters)] = 1
    return tensor

############# Functions for training the SRN - you'll modify these #############


def train_srn(training_set, testing_set, nepochs, rnn, optimizer, criterion,
    eval_single_output_fn=None, verbose = False):
    '''
    Trains the given SRN using criterion to calculate the loss and optimizer
    to adjust the weights. Prints out the training loss, training accuracy,
    and test accuracy every 20 epochs, as well as after the first and last
    epoch.
    For the first part of implementing train_srn, you won't use the input
    eval_single_output_fn - you'll modify train_srn later in the assignment to
    use this input.
    '''
    all_letters = 'BTSXPVE'
    for i in range(nepochs):
        ts = np.random.permutation(training_set)

        # make it into a tensor
        inputt = [seq_to_tensor(c, all_letters) for c in ts]

        loss = 0
        for t in inputt:
            loss += train_pattern(t, rnn, criterion, optimizer)

        if (i % 20 == 0) or (i == (nepochs - 1)):
            print('Epoch:', i)
            print('\tTotal Loss:\t', loss)

            if (eval_single_output_fn is not None):
                if i == (nepochs -1):
                    print('\tTraining Accuracy:\t', round(eval_set(ts, rnn, all_letters, eval_single_output_fn, verbose)*100,4), '%', sep='')
                    print('\tTesting Accuracy:\t', round(eval_set(testing_set, rnn, all_letters, eval_single_output_fn)*100,4), '%', sep='')
                else:
                    print('\tTraining Accuracy:\t', round(eval_set(ts, rnn, all_letters, eval_single_output_fn)*100,4),'%',sep='')
                    print('\tTesting Accuracy:\t', round(eval_set(testing_set, rnn, all_letters, eval_single_output_fn)*100,4),'%',sep='')
        # calculate total loss

        # calculate avg

    # TODO: You'll implement this function.
    # It should work as follows:
    # - It should go through the training_set nepochs times.
    # - In each epoch, it should go through the training_set once, in a random
    # order (changing the random order every time - this helps prevent it from
    # falling into a local optimum; you may find np.random.permutation helpful-
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html)
    # - For each pattern in the training set, you should run train_pattern.
    # This function adjusts the weights based on that pattern and returns the loss
    # of the network for that pattern - you implemented it in the previous part.
    # - Every 20 epochs (as well as after the first and last epoch) print out
    # the network's total loss for this epoch.
    # pass # delete this line when you write your solution


def train_pattern(seq_tensor, rnn, criterion, optimizer):
    '''
    Trains the given rnn on the pattern represented by
    seq_tensor (a 3D PyTorch tensor, where each item
    represents a one-hot encoding of a given symbol).
    criterion is a function that takes the network output for
    one symbol and a tensor representing the desired input as
    an index and returns the loss for that step.
    optimizer is an optimization function that will adjust
    the weights
    '''
    # First, we need to set the SRN in train mode and clear any previous gradients
    rnn.train()
    rnn.zero_grad()

    # Here's the part you'll implement: Running the SRN on
    # the symbols in seq_tensor and accumulating the loss.
    # To do this, you'll need to pass the inputs (which are
    # already one-hot encoded in the appropriate format for
    # the SRN) to the network one-by-one, implicitly calling
    # the forward function with the input and the hidden activation
    # as discussed in the comments of the SRN class (defined
    # in the notebook). Take a look at that class to figure out
    # what to use as the hidden activation for the first symbol.
    # loss is initialized for you, and you can add the output of
    # the criterion function to it with every additional symbol.
    # Note that criterion takes as parameters the output from the
    # network and the desired output according to seq_tensor.
    # The desired output is represented as a number rather than
    # a one-hot vector, so you'll want to use the function one_hot_to_index,
    # which takes in a one-hot vector and returns the corresponding
    # number.

    loss = 0
    # YOUR CODE HERE
    hidden = rnn.initHidden()
    inputt = torch.zeros(1, 7)
    for i in range(len(seq_tensor)):
        output, hidden = rnn(inputt, hidden)
        loss += criterion(output, one_hot_to_index(seq_tensor[i]))
        inputt = seq_tensor[i]

    # After you've processed the sequence, the error information
    # is used to calculate the gradients and adjust the weights.
    # (You don't need to modify anything else in this function)
    loss.backward()  # Calculate the gradients
    optimizer.step()  # Adjust the weights based on the gradients
    # Return the per symbol loss for this sequence
    return loss.data.numpy() / float(seq_tensor.size()[0] - 1)


############# Functions for evaluating the SRN  #############
####You'll complete the eval_pattern function and you'll ####
####call eval_set from your (revised) train_srn. You     ####
####won't modify eval_single_output_reber. You're welcome####
####to add additional evaluation fns to help you explore ####
####what the srn has learned.                            ####
#############################################################

def eval_set(list_seq, rnn, all_letters, eval_single_output_fn, verbose = False):
    '''
    Calculates and returns the accuracy of the SRN (named rnn)
    when predicting next letters for the string in the list list_seq.
    A prediction is counted as correct if the letter with highest
    probability in the output is a possible next letter according to
    the grammar.

    all_letters: a string containing all possible symbols in the grammar; ordered
      in the same way as the input to the rnn (to allow converting to one-hot input)
    eval_single_output_fn: a function that takes three parameters - the output of the network
      for predicting the next symbol, the letters so far in the sequence, and all_letters -
      and returns True if the prediction is correct and False if the prediction is incorrect.
      (For example, if seq is "BPVPSE" and we're predicting the final symbol, then the
       letters so far in the sequence would be "BPVPS".)
    '''
    # The syntax below is a list comprehension - it says make a list of the result of the
    # eval_pattern call applied to every element in list_seq. Ask in office hours or on
    # piazza if you have questions!
    #pattern_acc = [eval_pattern(seq, rnn, all_letters, eval_single_output_fn, verbose) for seq in list_seq]
    pattern_acc = [eval_pattern(seq, rnn, all_letters, eval_single_output_fn) for seq in list_seq[:-1]]
    pattern_acc.append(eval_pattern(list_seq[-1], rnn, all_letters, eval_single_output_fn, verbose))
    return np.mean(pattern_acc)


def eval_pattern(seq, rnn, all_letters, eval_single_output_fn, verbose = False):
    '''
    Calculates the SRN's (named rnn) percentage correct when
    predicting next letters for the string seq.

    all_letters: a string containing all possible symbols in the grammar; ordered
      in the same way as the input to the rnn (to allow converting to one-hot input)
    eval_single_output_fn: a function that takes three parameters - the output of the network
      for predicting the next symbol, the letters so far in the sequence, and all_letters -
      and returns True if the prediction is correct and False if the prediction is incorrect.
      (For example, if seq is "BPVPSE" and we're predicting the final symbol, then the
       letters so far in the sequence would be "BPVPS".)
    '''
    if not all(c in all_letters for c in seq):
        raise Exception('Input sequence contains an invalid letter')
    # First, we set the SRN in eval mode - we're not going to adjust any weights
    rnn.eval()

    # Here's the part you'll implement: Running the SRN on
    # the symbols in seq_tensor and calculating the percent
    # that are correct.
    # This should look somewhat similar to your train_pattern,
    # but here, you're not accumulating the loss but instead
    # calculating percent correct (the percent of the network's
    # predictions that are correct). You'll call eval_single_output_fn
    # to calculate correctness for a single prediction.
    # To do this, you'll need to pass the inputs (which are
    # already one-hot encoded in the appropriate format for
    # the SRN) to the network one-by-one, implicitly calling
    # the forward function with the input and the hidden activation
    # as discussed in the comments of the SRN class (defined
    # in the notebook). Take a look at that class to figure out
    # what to use as the hidden activation for the first symbol.
    # loss is initialized for you, and you can add the output of
    # the criterion function to it with every additional symbol.
    # Note that criterion takes as parameters the output from the
    # network and the desired output according to seq_tensor.
    # The desired output is represented as a number rather than
    # a one-hot vector, so you'll want to use the function one_hot_to_index,
    # which takes in a one-hot vector and returns the corresponding
    # number.

    accuracy = []
    seq_tensor = seq_to_tensor(seq, all_letters) #turn into tensors
    hidden = rnn.initHidden()
    inputt = torch.zeros(1, 7)
    if verbose:
        print(seq)
    for i in range(len(seq_tensor)):
        output, hidden = rnn(inputt, hidden)
        accuracy.append(eval_single_output_fn(output, seq[:i], all_letters, verbose))
        inputt = seq_tensor[i]
    return(np.mean(accuracy))


def eval_single_output_reber(output, preceding_letters, all_letters, strings_possible, verbose=False):
    '''
    Calculates correctness for the Reber grammar. output is the output
    of an SRN for predicting the next letter in a sequence where the
    previous letters are preceding_letters (a string). strings_possible
    is all possible strings in the grammar. Returns True if the letter
    with highest probability in output is a possible next letter in
    the sequence given the preceding letters. "possible next letter"
    is approximated as occurring in the list of all possible strings.
    '''
    widx=output.data.topk(1)[1].numpy()[0][0]
    prediction=preceding_letters + all_letters[widx]
    context=[s[:len(prediction)] for s in strings_possible]
    if verbose:
        print('\t',preceding_letters, ' ', all_letters[widx], prediction in context) #print prediction
    return prediction in context




############# Functions for visualizing the SRN #############
####You'll modify eval_viz but can ignore the rest       ####
#############################################################

def eval_viz(seq, rnn, all_letters):
    '''
    Takes in a string of symbols (seq), an SRN (rnn), and a string
    with all_letters (in the same order as the one-hot encoding).
    Evaluates the sequence with the SRN and displays the input symbol,
    hidden activations and output probabilities at each step.
    '''
    # The line below checks that the input sequence contains only allowed letters
    if not all(c in all_letters for c in seq):
        raise Exception('Input sequence contains an invalid letter')

    rnn.eval()  # Switch the SRN into evaluate (rather than train) mode
    # TODO: You'll implement the remainder of this function
    # You'll follow a similar idea to train_pattern, but now you're not
    # changing the weights but instead displaying what happens with each
    # symbol. Call display_srn_output (defined below) to do the actual
    # printing. Are there any symbols or parts of the sequence where you
    # shouldn't print the output probabilities because the network is
    # done predicting?
    
    
    seq_tensor = seq_to_tensor(seq, all_letters) #turn into tensors
    hidden = rnn.initHidden()
    inputt = torch.zeros(1, 7)
    for i in range(len(seq_tensor)):
        output, hidden = rnn(inputt, hidden)
        inputt = seq_tensor[i]
        
        #printing stuff
        widx=output.data.topk(1)[1].numpy()[0][0]
        predicted = all_letters[widx]
        output_print = [round(x,2) for x in output[0].tolist()]
        input_print = [round(x,2) for x in seq_tensor[i][0].tolist()]
        hidden_print = [round(x,2) for x in hidden[0].tolist()]
        if i==0:
            print('Current Input Symbol:', "" )
        else:
            print('Current Input Symbol:', seq[i-1])
        print('  Input:', end = '\t')
        for x in range(len(all_letters)):
            print(all_letters[x],':', input_print[x],end=' ',sep='')
        print('\nPredicted Next Symbol:',  predicted)
        print('  Hidden:', end='\t')
        [print(round(h,2), end = ' ') for h in hidden_print]
        print('\n  Output:',end='\t')
        for x in range(len(all_letters)):
            print(all_letters[x],':',round(output_print[x],2),end=' ', sep='')
        print()
        print()
    pass  # delete this line and replace with your code

def display_srn_output(input, output, hidden, all_letters):
    '''
    Takes the input, hidden, and output tensors, as well as a string
    with all_letters (in the same order as the one-hot encoding), and
    prints out the input symbol and its one hot encoding, the activations
    on the hidden units, and the probabilities for each output symbol.
    (You'll call but not modify this function)
    '''
    v_extract=lambda x: x.data.numpy().flatten()
    output_prob=np.exp(v_extract(output))
    print('  Input:', end=' ')
    display_io_v(v_extract(input), all_letters)
    print('Predicted next symbol:')
    print('  Hidden:', end=' ')
    display_v(v_extract(hidden))
    print('  Output:', end=' ')
    display_io_v(output_prob, all_letters)
    print(" ")

def display_io_v(v, all_letters):
    '''
    (You can ignore this function)
    '''
    assert len(v) == len(all_letters)
    for idx, c in enumerate(all_letters):
        print('%s:%2.2f' % (c, v[idx]), end=' ')
    print('')

def display_v(v):
    '''
    (You can ignore this function)
    '''
    for vi in v:
        print('%2.2f' % (vi), end=' ')
    print('')
