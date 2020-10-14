# Vulgaris
Project to analyze Italian Diachronic Language Varieties

## Cite

```
@misc{zugarini2020vulgaris,
      title={Vulgaris: Analysis of a Corpus for Middle-Age Varieties of Italian Language},
      author={Andrea Zugarini and Matteo Tiezzi and Marco Maggini},
      year={2020},
      eprint={2010.05993},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Download Script
*Disclaimer:* we retrieved and analyzed the data from Biblioteca Italiana solely for personal and academic non-commercial purposes.
To replicate our analyzes and ease the diachronic language research,
we provide the following script that retrieves and organizes the corpus in a convenient structure.

```
python vulgaris_project.py
```

By running that script, you declare to respect the following copyright of Biblioteca Italiana:
[Creative Common License](http://creativecommons.org/licenses/by-nc-nd/2.0/it/)
[![Creative Commons](https://i.creativecommons.org/l/by-nc-nd/2.0/it/88x31.png)](http://creativecommons.org/licenses/by-nc-nd/2.0/it/)

## Perplexity-base Analysis
First you should retrieve the data.

```
python char_diachronic_lm_exp.py path/to/vulgaris.csv
```