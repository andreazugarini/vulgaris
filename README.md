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
By running the following script, you declare to respect the following copyright by Biblioteca Italiana.

```
python vulgaris_project.py
```
## Perplexity-base Analysis
First you should retrieve the data.

```
python char_diachronic_lm_exp.py path/to/vulgaris.csv
```