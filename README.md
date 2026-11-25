<h1 align="center">Segment tracts into multiple bundles using QuickBundles</h1>

<p align="center">
  A new white matter classification in which every tract is split into numbered sub-bundles, clustered with DIPY QuickBundles.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.438"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.438-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="MIT license"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Splits each tract of a white matter classification (wmc) into a chosen number of sub-bundles with DIPY QuickBundles and outputs a new wmc classification.**

For each tract in the input classification, runs QuickBundles on its streamlines and adjusts the distance threshold until the requested number of clusters is reached, then writes a new wmc in which every tract is replaced by its numbered sub-bundles. Use it to study within-tract organisation; the tractogram must be the one the classification indexes into.

| | |
|---|---|
| **Inputs** | `neuro/dwi` · `neuro/track/tck` · `neuro/wmc` |
| **Outputs** | `neuro/wmc` |
| **Container** | `brainlife/pyafq:dev` |
| **Resources** | 2 CPUs · 2 h walltime |
| **Run it** | [doi.org/10.25663/brainlife.app.438](https://doi.org/10.25663/brainlife.app.438) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `dwi` | `neuro/dwi` | yes | Source dwi for tractogram. Only used in loading |
| `tractogram` | `neuro/track/tck` | yes | Whole-brain tractogram the classification indexes into. |
| `classification` | `neuro/wmc` | yes | White matter classification (wmc) whose tracts are split into sub-bundles. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | New white matter classification with each input tract split into number_bundles QuickBundles sub-bundles. |

<details>
<summary><b>Parameters</b> (1)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `number_bundles` | string | `3` | Number of sub-bundles to split each tract into. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.438](https://doi.org/10.25663/brainlife.app.438), choose a white matter classification (`neuro/wmc`), the whole-brain tractogram (`neuro/track/tck`) it indexes into and the DWI the tractogram was made from (used only as the reference when loading the tractogram), set `number_bundles` and press **Execute**. The App archives a new `neuro/wmc` dataset tagged `quickbundles`.

For each tract, the App runs QuickBundles on the tract's streamlines, starting at a 1 mm distance threshold and adjusting it until exactly `number_bundles` clusters are found. Use it to study the organisation within a tract.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/). The App runs Python 3 with DIPY, nibabel, SciPy and matplotlib from the `brainlife/pyafq:dev` container.

1. Clone this repository.
2. Create `config.json` with the paths to your inputs (see `config.json_sample`):

   ```json
   {
       "number_bundles": "3",
       "dwi": "./data/dwi/dwi.nii.gz",
       "tractogram": "./data/track/track.tck",
       "classification": "./data/wmc/classification.mat"
   }
   ```

3. Run the App:

   ```bash
   ./main
   ```

### Sample data

Download sample datasets from brainlife.io with the [brainlife CLI](https://github.com/brainlife/cli):

```bash
npm install -g brainlife
bl login
mkdir input
bl dataset download <dataset-id>
```

## Outputs

```text
wmc/
├── classification.mat   `names` (<tract>_1, <tract>_2, …) and per-streamline `index`
└── tracts/              one JSON per sub-bundle (coordinates rounded to two decimals) and tracts.json
```

## Troubleshooting

- **`failed`**: `wmc/classification.mat` was not produced. Check the log of `initialSegmentation.py`, and that the tractogram is the one the classification indexes into.
- **`number_bundles` other than 3**: the code names the sub-bundles `<tract>_1` to `<tract>_3` regardless of `number_bundles`, so other values may fail or mislabel bundles.

## Citing

> [!TIP]
> If you use this App in your work, please cite the App, the software it builds on and brainlife.io (below). GitHub's **Cite this repository** button gives the App citation in APA and BibTeX.

- Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I and Dipy Contributors. DIPY, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8:8 (2014).
- Segment tracts into multiple bundles using QuickBundles. brainlife.io App. [doi:10.25663/brainlife.app.438](https://doi.org/10.25663/brainlife.app.438)
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Kruper et al. Evaluating the Reliability of Human Brain White Matter Tractometry. *Aperture Neuro* 2021 (2021). [doi:10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669](https://doi.org/10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Brad Caron

**Contributors:** Soichi Hayashi

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io is publicly funded, and acknowledging that support helps keep it running. This App was developed with support from Wellcome, the U.S. National Institutes of Health (including the BRAIN Initiative BRAIN CONNECTS program: The Axonal Projectome EXchange, U24NS140384, and the Center for Mesoscale Connectomics, UM1NS132207) and the U.S. National Science Foundation.</sub>

## License

MIT, Copyright (c) 2020 brainlife.io, The University of Texas at Austin and Indiana University. DIPY itself is distributed under its own license.
