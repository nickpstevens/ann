import numpy as np
from scipy.special import expit
from random import shuffle, seed, randint
from collections import Counter
from mldata import *

"""
Written by Nick Stevens
10/9/2016
"""

# Useful constants
CLASS_LABEL = -1


class ANN(object):
    LEARNING_RATE = 0.01

    def __init__(self, training_set, validation_set, num_hidden_units, weight_decay_coeff):
        # Set up the training and validation sets
        self.full_training_set = np.array(training_set.to_float(), ndmin=2)
        self.full_validation_set = np.array(validation_set.to_float(), ndmin=2)
        np.random.seed(12345)
        # Shuffle the sets so that they are not ordered by class label
        np.random.shuffle(self.full_training_set)
        np.random.shuffle(self.full_validation_set)
        self.training_labels = self.full_training_set[:, [CLASS_LABEL]]
        self.training_examples = self.full_training_set[:, :CLASS_LABEL]
        self.validation_labels = self.full_validation_set[:, [CLASS_LABEL]]
        self.validation_examples = self.full_validation_set[:, :CLASS_LABEL]
        # Standardize matrices with respect to each feature
        self.training_examples = self.standardize(self.training_examples)
        self.validation_examples = self.standardize(self.validation_examples)
        (self.num_training_examples, self.num_features) = self.training_examples.shape
        # Make sure training set and validation set are compatible
        assert self.num_features == self.validation_examples.shape[1]
        # Hidden Layer setup
        self.hidden_weights = np.random.uniform(-0.1, 0.1, (self.num_features, num_hidden_units))
        self.hidden_inputs = None
        self.hidden_outputs = None
        # Output Layer setup
        self.output_weights = np.random.uniform(-0.1, 0.1, (num_hidden_units, 1))
        self.output_inputs = None
        self.output_sigmoids = None
        self.output_labels = np.empty(self.training_labels.shape)
        # Other variables
        self.weight_decay_coeff = weight_decay_coeff

    def standardize(self, example_set):
        """
        Replaces each feature value x in the example set with (x - mean(x)) / standard_deviation(x)
        Any NaN values caused by 0s in the standard deviation are just replaced with 0.
        Uses ddof=1 because this is calculating the sample standard deviation.
        """
        standardized = (example_set - np.mean(example_set, axis=0)) / np.std(example_set, axis=0, ddof=1)
        standardized = np.nan_to_num(standardized)
        return standardized

    def train(self, num_training_iters):
        if num_training_iters == 0:
            while not np.array_equal(self.output_labels, self.training_labels):
                self.stochastic_learning()
            print(self.output_labels)
            print(self.training_labels)
        else:
            for i in range(0, num_training_iters):
                self.stochastic_learning()

    def stochastic_learning(self):
        """
        For each example in the training set, feed it through the neural network and then use backpropagation to update
        the weights. Stochastic learning is used here instead of batch learning to avoid local minima, but also because
        I got tired of dealing with accumulated errors that were too big for the sigmoid function to handle.
        """
        for i in xrange(0, self.num_training_examples):
            actual_label = np.array(self.training_labels[i], ndmin=2)
            example = np.array(self.training_examples[i, :], ndmin=2)
            self.feedforward(example, i)
            self.backpropagation(actual_label, example)
        print('Iteration accuracy:\t' + str(np.sum(self.output_labels == self.training_labels) /
                                            float(self.num_training_examples)))

    def feedforward(self, examples, index):
        # Feed examples through Hidden Layer
        self.hidden_inputs = np.dot(examples, self.hidden_weights)
        self.hidden_outputs = self.sigmoid(self.hidden_inputs)
        # Feed examples through Output Layer
        self.output_inputs = np.dot(self.hidden_outputs, self.output_weights)
        self.output_sigmoids = self.sigmoid(self.output_inputs)
        new_label = self.binary_values(self.output_sigmoids)
        np.put(self.output_labels, index, new_label)

    def backpropagation(self, actual_label, example):
        output_dl_dw = self.calc_output_dl_dw(actual_label)
        hidden_dl_dw = self.calc_hidden_dl_dw(example, output_dl_dw)
        self.update_weights(output_dl_dw, hidden_dl_dw)

    def update_weights(self, output_dl_dw, hidden_dl_dw):
        self.output_weights -= (self.LEARNING_RATE * (output_dl_dw + self.weight_decay_coeff * self.output_weights))
        self.hidden_weights -= (self.LEARNING_RATE * (hidden_dl_dw + self.weight_decay_coeff * self.hidden_weights))

    def calc_output_dl_dw(self, actual_label):
        """
        Calculates the loss due to the output-layer weights between the output unit and the hidden-layer outputs.
        """
        subtracted_term = self.output_sigmoids - actual_label
        d_sigmoid = self.d_sigmoid(self.output_inputs)
        d_sigmoid_times_inputs = np.dot(self.hidden_outputs.T, d_sigmoid)
        dl_dw = np.dot(d_sigmoid_times_inputs, subtracted_term)
        return dl_dw

    def calc_hidden_dl_dw(self, example, output_dl_dw):
        """
        Calculates the loss due to the hidden-layer weights between the hidden layer and the input units.
        This calculation is simplified by the fact that the only downstream unit is the single output unit.
        """
        d_sigmoid = self.d_sigmoid(self.hidden_inputs)
        d_sigmoid_times_examples = np.dot(example.T, d_sigmoid)
        downstream = np.apply_along_axis(self.calc_downstream_component, 1, self.hidden_outputs, output_dl_dw)
        downstream_avg = np.average(downstream, axis=0)
        downstream_avg = np.reshape(downstream_avg, (downstream_avg.shape[0], -1))
        dl_dw = np.multiply(d_sigmoid_times_examples, downstream_avg.T)
        return dl_dw

    def calc_downstream_component(self, example, output_dl_dw):
        return output_dl_dw.flatten() * np.divide(self.output_weights.flatten(), example)

    def sigmoid(self, x):
        sigmoid = np.copy(x)
        sigmoid[sigmoid < -709] = -709  # Standardize values too large for expit()
        sigmoid[sigmoid > 709] = 709
        sigmoid = expit(sigmoid)  # Efficient sigmoid calculation from scipy
        return sigmoid

    def d_sigmoid(self, x):
        """
        The derivative of the sigmoid function
        """
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    def binary_values(self, x):
        bin_x = np.copy(x)
        bin_x[bin_x > 0.5] = 1
        bin_x[bin_x <= 0.5] = 0
        return bin_x

    def evaluate(self):
        """
        Feed the validation set through the network
        """
        num_examples = len(self.validation_examples)
        for i in xrange(0, num_examples):
            example = np.array(self.validation_examples[i, :], ndmin=2)
            self.feedforward(example, [i])
        return self.validation_labels, self.output_labels[:num_examples]


def main(options):
    assert options is not None
    assert len(options) == 5
    file_base = options[0]
    example_set = parse_c45(file_base)
    schema = example_set.schema

    default_cv_option = 0
    default_num_hidden_units = 20
    default_weight_decay_coeff = 0.01
    default_num_training_iters = 0

    # If 0, use cross-validation. If 1, run algorithm on full sample.
    cv_option = (1 if options[1] == '1' else default_cv_option)
    try:
        num_hidden_units = (int(options[2]) if int(options[2]) > 0 else default_num_hidden_units)
    except ValueError:
        num_hidden_units = default_num_hidden_units
    try:
        weight_decay_coeff = float(options[3])
    except ValueError:
        weight_decay_coeff = default_weight_decay_coeff
    try:
        num_training_iters = (int(options[4]) if int(options[4]) > 0 else default_num_training_iters)
    except ValueError:
        num_training_iters = default_num_training_iters

    if cv_option == 1:
        accuracy, accuracy_std, precision, precision_std, recall, recall_std, area_under_roc \
            = run(example_set, example_set, num_hidden_units, weight_decay_coeff, num_training_iters)
        print('Area under ROC:\t' + str("%0.6f" % area_under_roc) + '\n')
    else:
        fold_set = k_folds_stratified(example_set, schema, 5)
        accuracy, accuracy_std, precision, precision_std, recall, recall_std \
            = run_cross_validation(fold_set, schema, num_hidden_units, weight_decay_coeff, num_training_iters)

    print('Accuracy:\t' + str("%0.6f" % accuracy) + '\t' + str("%0.6f" % accuracy_std))
    print('Precision:\t' + str("%0.6f" % precision) + '\t' + str("%0.6f" % precision_std))
    print('Recall:\t\t' + str("%0.6f" % recall) + '\t' + str("%0.6f" % recall_std))


def k_folds_stratified(example_set, schema, k):
    seed(12345)
    shuffle(example_set)
    label_dist = Counter(ex[CLASS_LABEL] for ex in example_set)
    label_values = label_dist.keys()
    examples_with_label = [[] for x in xrange(len(label_values))]
    # Get the set of examples for each label
    for example in example_set:
        for label in label_values:
            if example[CLASS_LABEL] == label_values[label]:
                examples_with_label[label].append(example)
                break
    # Group examples by class label
    sorted_examples = []
    for example_subset in examples_with_label:
        sorted_examples += example_subset
    folds = [ExampleSet(schema) for x in xrange(k)]
    # Distribute sorted examples evenly amongst all k folds
    for i in xrange(0, len(sorted_examples)):
        assigned_fold = i % k
        folds[assigned_fold].append(sorted_examples[i])
    return folds


def run_cross_validation(fold_set, schema, num_hidden_units, weight_decay_coeff, num_training_iters):
    num_folds = len(fold_set)
    assert num_folds != 0
    avg_accuracy = 0.0
    avg_accuracy_std = 0.0
    avg_precision = 0.0
    avg_precision_std = 0.0
    avg_recall = 0.0
    avg_recall_std = 0.0
    for i in xrange(0, num_folds):
        validation_set = fold_set[i]
        training_set = ExampleSet(schema)
        for j in xrange(1, num_folds):
            k = (i + j) % num_folds
            for example in fold_set[k]:
                training_set.append(example)
        accuracy, accuracy_std, precision, precision_std, recall, recall_std, area_under_roc \
            = run(training_set, validation_set, num_hidden_units, weight_decay_coeff, num_training_iters)
        print('Fold ' + str(i+1))
        print('Area under ROC:\t' + str(area_under_roc) + '\n')
        avg_accuracy += (1.0/num_folds) * accuracy
        avg_accuracy_std += (1.0/num_folds) * accuracy_std
        avg_precision += (1.0/num_folds) * precision
        avg_precision_std += (1.0/num_folds) * precision_std
        avg_recall += (1.0/num_folds) * recall
        avg_recall_std += (1.0/num_folds) * recall_std
    return avg_accuracy, avg_accuracy_std, avg_precision, avg_precision_std, avg_recall, avg_recall_std


def run(training_set, validation_set, num_hidden_units, weight_decay_coeff, num_training_iters):
    assert isinstance(training_set, ExampleSet)
    print('Building ANN\n')
    ann = ANN(training_set, validation_set, num_hidden_units, weight_decay_coeff)
    print('Training ANN\n')
    ann.train(num_training_iters)
    print('Evaluating ANN performance\n')
    return evaluate_ann_performance(ann)


def evaluate_ann_performance(ann):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    actual_labels, assigned_labels = ann.evaluate()
    num_examples = len(actual_labels)
    label_pairs = zip(actual_labels, assigned_labels)
    for labels in label_pairs:
        if labels == (1.0, 1.0):
            true_positives += 1
        elif labels == (1.0, 0.0):
            false_negatives += 1
        elif labels == (0.0, 1.0):
            false_positives += 1
        elif labels == (0.0, 0.0):
            true_negatives += 1
    print('\tValidation Set Distribution')
    print('\tTP: ' + str(true_positives))
    print('\tTN: ' + str(true_negatives))
    print('\tFP: ' + str(false_positives))
    print('\tFN: ' + str(false_negatives) + '\n')
    accuracy = float(true_positives + true_negatives) / num_examples
    accuracy_std = 0.0  # FIXME
    try:
        precision = float(true_positives) / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
    precision_std = 0.0  # FIXME
    try:
        recall = float(true_positives) / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0
    recall_std = 0.0  # FIXME
    area_under_roc = 0.0  # FIXME
    return accuracy, accuracy_std, precision, precision_std, recall, recall_std, area_under_roc


if __name__ == "__main__":
    main(sys.argv[1:])
