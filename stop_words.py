import os

STOP_WORDS = "data/stop_words.txt"


# A function to get the stop words as a list.
def get_stop_words():
    stop_words = []
    with open(STOP_WORDS) as file:
        contents = file.readlines()
        for line in contents:
            stop_words.append(line.replace("\n", ""))

    return stop_words
