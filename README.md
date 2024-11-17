# Project Alpha
This is the source code for a research project testing the relationship between poetry and prose. The idea is that there is a fundamental disconnect in the way that these two types of text are created and structured, such that Large Language Models (LLMs) simply cannot predict poetry. In other words, poetry does not contain the repetitive word and language patterns of prose and thus cannot be predicted with the same degree of accuracy.

This repository contains some Python programs and batch scripts for testing these hypotheses using state-of-the-art LLM AI tools, including Word2Vec, Alpa, and BERT.

The file structure is as follows:
- `src/files` - contains some of the texts or text data used for the hypothesis testing
- `src/python` - contains the main source code of the project
- `src/scripts` - contains some post-processing tools for organizing & compiling the results of the main programs
- `src/file.sh` - script for running the file managing CLI tool; this interacts with & dynamically updates the content in `src/files`
- `src/run.sh` - script for executing the main Python program
