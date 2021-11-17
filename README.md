# lane-detection

## Description

## Installation

```
Tested on:

Anaconda 4.10.3
Python 3.8.12
OpenCV 4.0.1
```

## Example

Run code:

```
conda create --name venv python=3.8
conda activate venv
python lane_detection.py 
conda deactivate
```

## Datasets

| Dataset                                                             | clips     | Length/clip   | FPS   | Scenes |
| ------------------------------------------------------------------- | -------:  | :-----------: | :---: | :------: |
| [CULane](https://xingangpan.github.io/projects/CULane.html)         | 55h       | frame             | -     | normal, crowded, night, no line, shadow, arrow, dazzle light, curve,                                                                                                              crossroad |
| [BDD100k](https://bdd-data.berkeley.edu/)                           | #100k    | 40sec         | 30    | city & residential & highway / diverse weather conditions / different                                                                                                             times of the day |
| [TUSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) | #3626    | 1 sec         | 20    | highway / good and medium weather / day time |

