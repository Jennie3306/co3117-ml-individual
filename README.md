# CO3117 – Machine Learning – Individual Longitudinal Assignment (HK261)

**Student:** Le Nhu Nha Uyen – 2453402 · **Class:** CC0... · **Instructor:** Nguyen An Khuong

**One dataset, one use case, many models.** Task: predict a person's current physical activity
(6 classes) from smartphone inertial-sensor data — UCI HAR (see [`data/README.md`](data/README.md)).

- Weekly progress: [`PROGRESS.md`](PROGRESS.md)
- Model log: [`MODEL_LOG.md`](MODEL_LOG.md) · AI use: [`AI_USE.md`](AI_USE.md) · References: [`REFERENCES.md`](REFERENCES.md)
- Learning blog: [`docs/`](docs/index.md)

## Experimental protocol (frozen at tag `release-baseline`)
See [`src/config.py`](src/config.py) and the *Protocol* section of [`data/README.md`](data/README.md).

## Setup & reproduction
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download the data as described in data/README.md, then:
python -m pytest -q
python -m experiments.part1_pre_midterm.r0_explore
python -m experiments.part1_pre_midterm.r0_baseline
python -m experiments.part1_pre_midterm.r0_tree_curves
```
Tested environment: Python 3.13.15, operating system: Windows 11 (64-bit)
