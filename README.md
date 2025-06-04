# bibtex-formatter

## What is this repo?

Standardizing the increasingly complicated BibTeX format

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/bibtex-formatter.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

### Execution
```
$ python clean_bib.py input.bib -o output.bib
```
#### Argument
- `bibfile`
  - name of .bib file you want to format
    - e.g., `input.bib`
- `-o` | `--output`
  - specify the name of the output .bib file after formatting
    - e.g., `-o output.bib`
  - if not specified, the filename will be the original `bibfile` name with `_clean` appended to the end.
    - e.g., if you specify `reference.bib` as `bibfile`, then output file name is `reference_clean.bib`

## Contact

If you have any question, please feel free to contact: stamano@niid.go.jp
