from NB import Naive_Bayes


def main():
    use_default = input('Use default inputs (y/n)? ')

    if not use_default:
        train_spam = input('Name of spam training file: ')
        train_ham = input('Name of ham training file: ')
        test_spam = input('Name of spam testing file: ')
        test_ham = input('Name of ham testing file: ')
    else:
        train_spam = './data/train-spam.txt'
        train_ham = './data/train-ham.txt'
        test_spam = './data/test-spam.txt'
        test_ham = './data/test-ham.txt'

    with open('./data/stopwords.txt', "r") as file:
        stopwords = file.read()
    stopwords = stopwords.split('\n')

    nb = Naive_Bayes(stopwords=stopwords)
    nb.fit([train_spam, train_ham])
    nb.predict([test_spam, test_ham])

    print('Accuracy: {:.2f}'.format(nb.accuracy_score()))
    print('Precision: {:.2f}'.format(nb.precision()))
    print('Recall: {:.2f}'.format(nb.recall()))
    print('F1: {:.2f}'.format(nb.f_score()))


if __name__ == "__main__":
    main()
