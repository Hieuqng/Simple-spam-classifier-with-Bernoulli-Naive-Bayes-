import math


class Naive_Bayes():
    def __init__(self, stopwords=[]):
        self.stopwords = stopwords
        self._vocab_spam = {}
        self._vocab_ham = {}
        self._num_train_spam = 0
        self._num_train_ham = 0
        self._num_test_spam = 0
        self._num_test_ham = 0
        self._true_positive = 0
        self._true_negative = 0

    def fit(self, filenames, stopwords=[]):
        train_spam, self._num_train_spam = self._tokenizer(filenames[0])
        train_ham, self._num_train_ham = self._tokenizer(filenames[1])
        self._vocab_spam, self._vocab_ham = self._extend(train_spam, train_ham)

    def predict(self, filenames, stopwords=[], verbose=True):
        # Read in test files
        test_spam, self._num_test_spam = self._tokenizer(filenames[0], False)
        test_ham, self._num_test_ham = self._tokenizer(filenames[1], False)
        num_emails = self._num_train_spam + self._num_train_ham

        # Test spams emails
        for i in range(1, len(test_spam) + 1):
            # Calculate log probability of posteriors
            H_spam, match = self._total_log_proba(test_spam[i], self._vocab_spam,
                                                  self._num_train_spam, num_emails)
            H_ham, match = self._total_log_proba(test_spam[i], self._vocab_ham,
                                                 self._num_train_ham, num_emails)

            # Compare log probabilities
            if H_spam > H_ham:
                if verbose:
                    print('TEST {} {}/{} features true {:.3f} {:.3f} spam right'
                          .format(i, match, len(self._vocab_spam), H_spam, H_ham))
                    self._true_positive += 1
            else:
                if verbose:
                    print('TEST {} {}/{} features true {:.3f} {:.3f} ham wrong'
                          .format(i, match, len(self._vocab_spam), H_spam, H_ham))

        # Test ham emails
        for i in range(1, len(test_ham) + 1):
            # Calculate log probability of posteriors
            H_spam, match = self._total_log_proba(test_ham[i], self._vocab_spam,
                                                  self._num_train_spam, num_emails)
            H_ham, match = self._total_log_proba(test_ham[i], self._vocab_ham,
                                                 self._num_train_ham, num_emails)

            # Compare log probabilities
            if H_ham > H_spam:
                if verbose:
                    print('TEST {} {}/{} features true {:.3f} {:.3f} ham right'
                          .format(i, match, len(self._vocab_spam), H_spam, H_ham))
                    self._true_negative += 1
            else:
                if verbose:
                    print('TEST {} {}/{} features true {:.3f} {:.3f} spam wrong'
                          .format(i, match, len(self._vocab_spam), H_spam, H_ham))

        if verbose:
            print(f'Accuracy: {self._true_positive + self._true_negative}'
                  + f'/{self._num_test_ham + self._num_test_ham}')

    def accuracy_score(self):
        return (self._true_positive + self._true_negative)/(self._num_test_ham + self._num_test_ham)

    def recall(self):
        return self._true_positive / self._num_test_spam

    def precision(self):
        false_positive = self._num_test_ham - self._true_negative
        return self._true_positive / (self._true_positive + false_positive)

    def f_score(self, beta=1):
        return (1 + beta**2) * self.precision() * self.recall() / ((1 + beta**2) * self.precision() + self.recall())

    ''' Processing input files
    Input:
    - path to a txt file
    - is_train: optional, if we want to process a train or test set.

    Return: If train, vocab = {word: frequency}
            If test, vocab = {email_No: set_of_words}
    '''

    def _tokenizer(self, filename, is_train=True):
        file = open(filename, "r")
        lines = file.readlines()
        file.close()

        vocab = {}
        num_emails = 0
        email = []
        for l in lines:
            # Start of the (new) email
            if l == '<SUBJECT>\n':
                email = []
                num_emails += 1
                continue

            # End of the email
            if l in ['</BODY>\n', '</BODY>']:
                # If test, vocab = {email_No: set_of_words}
                if not is_train:
                    vocab[num_emails] = set(email)
                    continue

                # If train, vocab = {word: frequency}
                for word in set(email):
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            # Process each line in the email
            # Ignore tags
            if l in ['</SUBJECT>\n', '<BODY>\n', ]:
                continue

            # Do simple cleaning
            l = l.replace('\n', '')
            l = l.lower()

            # Get separated words for each email
            for word in l.split(' '):
                # Prune out very common words (if required)
                if word not in self.stopwords and word != '':
                    email.append(word)

        return (vocab, num_emails)

    '''
    Combine spam and ham vocab_dict, and aslo transform both:
    if a word is in one dict but not the other, add the pair {word: 0}
    to the other
    '''

    def _extend(self, d1, d2):
        new_d1 = {}
        new_d2 = {}
        all_words = set(d1.keys()).union(set(d2.keys()))

        for word in all_words:
            if word not in d1:
                new_d1[word] = 0
            else:
                new_d1[word] = d1[word]

        for word in all_words:
            if word not in d2:
                new_d2[word] = 0
            else:
                new_d2[word] = d2[word]

        return new_d1, new_d2

    '''
    Calculate the log probability with Laplace smoothing
    '''

    def _total_log_proba(self, text, vocab, num_class, num_total):
        total_log_proba = math.log(num_class / num_total)
        match = 0

        for word in vocab:
            if word in text:
                match += 1
                total_log_proba = total_log_proba \
                    + math.log((vocab[word] + 1) / (num_class + 2))
            else:
                total_log_proba = total_log_proba \
                    + math.log((num_class - vocab[word] + 1) / (num_class + 2))

        return total_log_proba, match
