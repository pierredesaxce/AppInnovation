from __future__ import unicode_literals, print_function, division

import unidecode
import string
import random
import re
import sys

from io import open
import glob
import os
import unicodedata
import string
import random

import time
import math

from os import listdir, path, makedirs, popen
from os.path import isdir, isfile, join, basename

import torch
import torch.nn as nn
from torch.autograd import Variable

import time, math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from argparse import ArgumentParser

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')

n_iters = 100000
all_losses = []
total_loss = 0  # Reset every plot_every iters

n_epochs = 200000
print_every = 10
plot_every = 10
hidden_size = 512
n_layers = 2
lr = 0.005
bidirectional = True

# Chargement du fichier de mots de passe (un mot de passe par ligne)
with open("data/Ashley-Madison.txt", "r") as file:
    # Filter passwords with max_len characters
    passwords = [line.strip() for line in file]

# Création d'un dictionnaire de caractères uniques
all_letters = ''.join(sorted(set("".join(passwords))))
print('all_letters: ', all_letters)
n_letters = len(all_letters) + 1  # Plus EOS marker


def findFiles(path): return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [line.strip() for line in some_file]


#filename = 'data/names/Russian.txt'
#filenameTrain = 'data/RussianTrain.txt'
#filenameTest = 'data/RussianTest.txt'


filename = 'Ashley-Madison.txt'
filenameTrain = 'data/Ashley-MadisonTrain.txt'
filenameTest = 'data/Ashley-MadisonTest.txt'



def getLines(f):
    lines = readLines(f)
    print('lines: ', len(lines), ' -> ', f)
    return lines


def split(rate, lines):
    categories = set(line[0] for line in lines)
    names_by_category = {category: [] for category in categories}

    for line in lines:
        category = line[0]
        names_by_category[category].append(line)

    print('split names: ', len(names_by_category))

    names_training = []
    names_testing = []

    for category, names_category in names_by_category.items():
        length = len(names_category)
        index = int(length * rate)

        training = names_category[:index]
        testing = names_category[index:]

        names_training.extend(training)
        names_testing.extend(testing)

    with open(filenameTrain, "w") as f:
        for names_line in names_training:
            f.write(names_line + "\n")

    with open(filenameTest, "w") as f:
        for names_line in names_testing:
            f.write(names_line + "\n")

    return names_training, names_testing



# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


def random_training_set(file):
    chunk = random_chunk(file)
    inp = char_tensor(chunk[:-1]).to(device)
    target = char_tensor(chunk[1:]).to(device)
    return inp, target


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTraining(lines):
    line = randomChoice(lines)
    return line


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)  # .long()
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(lines):
    line = randomTraining(lines)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor


def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = decoder.init_hidden()

    decoder.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i].to(device), hidden.to(device))
        l = criterion(output.to(device), target_line_tensor[i].to(device))
        loss += l

    loss.backward()

    # decoder_optimizer.step()
    for p in decoder.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.bidirectional = bidirectional
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,
                          bidirectional=self.bidirectional, batch_first=True)
        self.out = nn.Linear(self.num_directions * self.hidden_size, output_size)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)

        hidden_concatenated = hidden

        if self.bidirectional:
            hidden_concatenated = torch.cat((hidden[0], hidden[1]), 1)
        else:
            hidden_concatenated = hidden.squeeze(0)

        output = self.out(hidden_concatenated)

        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size)

    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

    def init_hidden_random(self):
        return torch.rand(self.num_directions, 1, self.hidden_size)
    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print('---')
        # print('input: ', input.size())
        # print('hidden: ', hidden.size())
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        # print('output: ', output.size())$
        # input:  torch.Size([1, 59])
        # hidden:  torch.Size([1, 128])
        # output:  torch.Size([1, 59])

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

    def init_hidden_random(self):
        return torch.rand(1, self.hidden_size)
    # return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))


def training(n_epochs, lines):
    print()
    print('-----------')
    print('|  TRAIN  |')
    print('-----------')
    print()

    start = time.time()
    all_losses = []
    total_loss = 0
    best_loss = 100
    print_every = n_epochs / 100

    for iter in range(1, n_epochs + 1):
        output, loss = train(*randomTrainingExample(lines))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), iter, iter / n_iters * 100, total_loss / iter, loss))


max_length = 20


def samples(start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(start_letter))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSinceStart(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


def progressPercent(totalNames, start, names, p, samplesGenerated):
    bar_len = 50
    filled_len = int(round(bar_len * names / float(totalNames)))
    percents = round(100.0 * names / float(totalNames), 1)
    nNames = int(p / 100 * totalNames)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write(
        '[%s] %s%s names founded among %d samples generated (%d of %d names) on %s (goal = %.1f%% = %d names)\r' % (
        bar, percents, '%', samplesGenerated, names, totalNames, timeSinceStart(start), p, nNames))
    sys.stdout.flush()


def progress(total, acc, start, epoch, l):
    bar_len = 50
    filled_len = int(round(bar_len * epoch / float(total)))
    percents = round(100.0 * epoch / float(total), 1)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s epoch: %d acc: %.3f %% and testing size = %d names => coverage of %.3f %% on %s \r' % (
    bar, percents, '%', epoch, (100 * acc / epoch), l, (100 * acc / l), timeSinceStart(start)))
    sys.stdout.flush()


def sample(decoder, start_letters='ABC'):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_hidden_random()

        if len(start_letters) > 1:
            for i in range(len(start_letters)):
                input = inputTensor(start_letters[i])
                # print(start_letters[i], ' ', hidden)
                output, hidden = decoder(input[0].to(device), hidden.to(device))

            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                return start_letters

            letter = all_letters[topi]
            input = inputTensor(letter)
        else:
            input = inputTensor(start_letters)

        output_name = start_letters

        for i in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


def testing(decoder, nb_samples, lineTest, percent):
    print()
    print('------------')
    print('|   TEST   |')
    print('------------')
    print()

    start = time.time()
    accuracy = 0
    predicted = "a"
    predicted_current = []

    if nb_samples > 0:

        for i in range(1, nb_samples + 1):
            # name max size ...
            nc = 1  # random.randint(1, max_length/2 - 1)
            # print('nc: ', nc, ' / ', max_length)

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = random.randint(0, len(all_letters) - 1)
                    starting_letters += all_letters[rc]
                # Ensure the length of starting_letters is between 1 and 3
                starting_letters = starting_letters[:random.randint(1, 3)]
                predicted = sample(decoder, starting_letters)

            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy = accuracy + 1
                print(starting_letters, ' -> ', predicted, ' accuracy: ', accuracy, ' / ', i , ' = ', (100 * accuracy/i) , ' % and testing corpus contains ', len(lineTest), ' names => coverage of ', (100 * accuracy/len(lineTest)), ' %')

            progress(total=nb_samples, acc=accuracy, start=start, epoch=i, l=len(lineTest))

        accuracy = 100 * accuracy / nb_samples

        print('Accuracy: ', accuracy, '%')

    else:
        i = 0
        l = len(lineTest)
        p = int(percent / 100 * l)
        while accuracy < p:
            nc = random.randint(1, int(max_length / 2 - 1))

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = random.randint(0, len(all_letters) - 1)
                    starting_letters = starting_letters + all_letters[rc]

                predicted = sample(decoder, starting_letters).lower()

            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy = accuracy + 1
            # print(starting_letters, ' -> ', predicted, ' accuracy: ', accuracy, ' founded over ', l, ' names in ', timeSinceStart(start), ' s and ', i, ' epochs => coverage of ', percent, ' % of total names names = ', p)

            i = i + 1
            progressPercent(totalNames=l, start=start, names=accuracy, p=percent, samplesGenerated=i)

        print(percent + ' % of all names (', len(lineTest), ') reached in ', i, 'iterations (', timeSinceStart(start),
              ' s)...')


def evaluating(decoder):
    print()
    print('------------')
    print('|   EVAL   |')
    print('------------')
    print()

    try:
        while True:
            print('Enter a starting two or tree charachters but less than ', (2 * max_length), ' charachters: ')
            starting_letters = input()
            print()
            if len(starting_letters) > 0 and len(starting_letters) < (2 * max_length):
                print('Generated up to ', max_length, 'charcaters: ')
                predicted = sample(decoder, starting_letters)
                print(predicted)
            else:
                print(starting_letters, ' length < 1 or > ', (2 * max_length))
            print('------------')
            print()

    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate evaluating process")
        print('------------')


def getMeanSize(listData):
    mean = 0
    for word in listData:
        mean = mean + len(word)

    return int(mean / len(listData))


if __name__ == '__main__':

    parser = ArgumentParser()
    #
    parser.add_argument("-d", "--trainingData", default="data/shakespeare.txt", type=str,
                        help="trainingData [path/to/the/data]")
    parser.add_argument("-te", "--trainEval", default='train', type=str, help="trainEval [train, eval, test]")
    #
    parser.add_argument("-r", "--run", default="rnnGeneration", type=str, help="name of the model saved file")
    # parser.add_argument("-mt", "--modelTraining", default='models', type=str, help="Path of the model to save (train) [path/to/the/model]")
    # parser.add_argument("-me", "--modelEval", default='models', type=str, help="Name of the model to load (eval) [path/to/the/model]")
    parser.add_argument("-m", "--model", default='models/rnn_v3.pt', type=str,
                        help="Path of the model to save for trainingof to load for evaluating/testing (eval/test) [path/to/the/model]")
    #
    parser.add_argument('--n', default=100000, type=int,
                        help="number of samples to generate [< 1000]. If < 0, the algorithm will provide names till it reaches the percent (see --p option)")
    parser.add_argument('--ml', default=10, type=int,
                        help="number of characters to generate for each name [default =10]. if < 0 => the number of chars = mean(training set)")
    parser.add_argument('--s', default=0.7, type=int,
                        help="percent of the dataset devoted for training [default =70% and therefore testing =30%]")
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool, help="Bidirectionnal model [default True]")
    parser.add_argument('--max_epochs', default=200000, type=int)
    parser.add_argument('-p', '--percent', default=15, type=float,
                        help="percent (number between 1 and 100) of the total names to find (test) [default 15%]")
    #
    args = parser.parse_args()
    #
    repData = args.trainingData  # "data/out/text10.txt"
    # repData = "data/shakespeare.txt"

    file = unidecode.unidecode(open(repData).read())
    file_len = len(file)
    bidirectional = args.bidirectional
    lines = getLines(filename)
    train_set, test_set = split(args.s, lines)

    print('filenameTrain: ', filenameTrain)
    lineTraining = getLines(filenameTrain)
    lineTest = getLines(filenameTest)

    print('lineTraining: ', len(lineTraining))
    print('lineTest: ', len(lineTest))

    if args.ml > 0:
        max_length = args.ml
    else:
        max_length = getMeanSize(lineTraining)

    decoder = RNNLight(n_letters, 128, n_letters).to(
        device)  # RNN(n_characters, args.hidden_size, n_characters, args.num_layers).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    print('decoder: ', decoder)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs = args.max_epochs

    modelFile = args.run + "_" + str(args.num_layers) + "_" + str(args.hidden_size) + ".pt"

    if not path.exists(args.model):
        makedirs(basename(args.model))

    #########
    # TRAIN #
    #########
    if args.trainEval == 'train':
        decoder.train()
        training(n_epochs, lineTraining)
        torch.save(decoder, args.model)
        print('Model saved in: ', args.model)
    #########
    # EVAL  #
    #########
    elif args.trainEval == 'eval':
        decoder.eval()
        decoder = torch.load(args.model)
        decoder.eval().to(device)
        evaluating(decoder)
    #########
    # TEST  #
    #########
    elif args.trainEval == 'test':
        decoder.eval()
        decoder = torch.load(args.model)
        decoder.eval().to(device)
        testing(decoder, args.n, lineTest, args.percent)
    else:
        print('Choose trainEval option (--trainEval train/eval/test')
