# Hidden-Markov-Model-Part-of-Speech-Tagger

This project implements a Part-of-Speech (POS) tagger using Hidden Markov Models (HMM) with the Viterbi algorithm. It includes multiple implementations with increasing sophistication in handling unknown words and improving accuracy.

## Overview

The system uses HMM to predict the most likely sequence of POS tags for a given sentence. It includes three main implementations:

1. Basic Viterbi implementation (not included in repository)
2. Improved Viterbi with better Laplace smoothing for unseen words
3. Enhanced Viterbi with suffix/prefix handling and word classification

## Features

- **Multiple Viterbi Implementations**: Different versions of the algorithm with progressive improvements
- **Unknown Word Handling**: Sophisticated handling of unseen words using:
  - Laplace smoothing
  - Hapax legomena probability estimation
  - Word classification based on length and suffixes
- **Performance Metrics**: Evaluation includes:
  - Overall accuracy
  - Accuracy on words with multiple tags
  - Accuracy on unseen words
  - Top K correct and incorrect predictions

## Technical Details

### Word Classification Categories
- `NUM`: Words containing only numbers
- `TINY`: Words less than 4 characters
- `SHORT_S`: Words between 4-9 characters ending with 's'
- `SHORT`: Words between 4-9 characters not ending with 's'
- `LONG_S`: Words 10+ characters ending with 's'
- `LONG`: Words 10+ characters not ending with 's'

### Components

- `mp8.py`: Main application file
- `utils.py`: Utility functions for data loading and evaluation
- `viterbi_2.py`: Implementation with improved Laplace smoothing
- `viterbi_3.py`: Enhanced implementation with word classification

## Usage

Run the tagger using:

```bash
python mp8.py --train [training_file] --test [test_file] --algorithm [algorithm_name]
```

Arguments:
- `--train`: Path to training data file
- `--test`: Path to test data file
- `--algorithm`: Algorithm to use (viterbi_2 or viterbi_3)

## Data Format

The input data should be a text file with each line containing word-tag pairs in the format:
```
word1=tag1 word2=tag2 word3=tag3 ...
```

## Performance Metrics

The system outputs:
- Overall accuracy
- Accuracy on words with multiple possible tags
- Accuracy on unseen words
- Top K correct and incorrect word-tag predictions

## Requirements

- Python 3.x
- No external libraries required

## Implementation Details

### Viterbi 2
- Implements improved Laplace smoothing
- Uses hapax legomena analysis for better unknown word handling
- Maintains separate emission and transition probabilities

### Viterbi 3
- Adds sophisticated word classification
- Enhanced handling of suffixes and prefixes
- Improved probability estimation for unknown words
- More robust smoothing techniques

## License

[Include your chosen license here]

## Contributing

[Include contribution guidelines if you want to accept contributions]
