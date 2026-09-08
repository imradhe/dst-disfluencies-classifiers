# Results

## Task: `fluent_vs_I`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc_sdc_mod_k7` | `dnn` | 0.948 | 0.576 | 0.948 | 0.960 | 0.768 | 0.984 |
| `mfcc45` | `dnn` | 0.924 | 0.536 | 0.924 | 0.947 | 0.671 | 0.984 |
| `mfcc_sdc_mod_k7` | `rf` | 0.881 | 0.517 | 0.881 | 0.923 | 0.734 | 0.984 |
| `mfcc45` | `rf` | 0.875 | 0.512 | 0.875 | 0.919 | 0.710 | 0.984 |

## Task: `fluent_vs_P`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc_sdc_mod_k7` | `rf` | 0.950 | 0.487 | 0.950 | 0.974 | 0.472 | 1.000 |
| `mfcc45` | `rf` | 0.935 | 0.483 | 0.935 | 0.966 | 0.326 | 1.000 |
| `mfcc_sdc_mod_k7` | `dnn` | 0.902 | 0.475 | 0.902 | 0.948 | 0.631 | 1.000 |
| `mfcc45` | `dnn` | 0.894 | 0.473 | 0.894 | 0.944 | 0.615 | 1.000 |

## Task: `fluent_vs_PR`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc_sdc_mod_k7` | `dnn` | 0.908 | 0.489 | 0.908 | 0.947 | 0.711 | 0.995 |
| `mfcc45` | `dnn` | 0.900 | 0.483 | 0.900 | 0.942 | 0.618 | 0.995 |
| `mfcc_sdc_mod_k7` | `rf` | 0.869 | 0.477 | 0.869 | 0.925 | 0.755 | 0.995 |
| `mfcc45` | `rf` | 0.841 | 0.469 | 0.841 | 0.909 | 0.700 | 0.995 |

## Task: `fluent_vs_PWR`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc45` | `dnn` | 0.911 | 0.491 | 0.911 | 0.940 | 0.526 | 0.985 |
| `mfcc_sdc_mod_k7` | `dnn` | 0.881 | 0.483 | 0.881 | 0.923 | 0.514 | 0.985 |
| `mfcc45` | `rf` | 0.738 | 0.443 | 0.738 | 0.837 | 0.554 | 0.985 |
| `mfcc_sdc_mod_k7` | `rf` | 0.732 | 0.439 | 0.732 | 0.832 | 0.530 | 0.985 |

## Task: `fluent_vs_PhR`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc45` | `dnn` | 0.913 | 0.489 | 0.913 | 0.928 | 0.493 | 0.972 |
| `mfcc_sdc_mod_k7` | `dnn` | 0.904 | 0.484 | 0.904 | 0.924 | 0.447 | 0.972 |
| `mfcc_sdc_mod_k7` | `rf` | 0.843 | 0.467 | 0.843 | 0.890 | 0.415 | 0.972 |
| `mfcc45` | `rf` | 0.828 | 0.463 | 0.828 | 0.881 | 0.449 | 0.972 |

## Task: `fluent_vs_WR`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc45` | `dnn` | 0.939 | 0.498 | 0.939 | 0.946 | 0.515 | 0.977 |
| `mfcc_sdc_mod_k7` | `dnn` | 0.945 | 0.495 | 0.945 | 0.949 | 0.459 | 0.977 |
| `mfcc45` | `rf` | 0.805 | 0.463 | 0.805 | 0.871 | 0.465 | 0.977 |
| `mfcc_sdc_mod_k7` | `rf` | 0.814 | 0.461 | 0.814 | 0.877 | 0.421 | 0.977 |

## Task: `fluent_vs_disfluent`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc_sdc_mod_k7` | `dnn` | 0.882 | 0.528 | 0.882 | 0.865 | 0.510 | 0.912 |
| `mfcc45` | `dnn` | 0.867 | 0.521 | 0.867 | 0.857 | 0.523 | 0.912 |
| `mfcc45` | `rf` | 0.731 | 0.494 | 0.731 | 0.780 | 0.518 | 0.912 |
| `mfcc_sdc_mod_k7` | `rf` | 0.734 | 0.490 | 0.734 | 0.781 | 0.496 | 0.912 |

## Task: `multiclass`

| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| `mfcc45` | `dnn` | 0.379 | 0.112 | 0.379 | 0.514 |  | 0.912 |
| `mfcc_sdc_mod_k7` | `dnn` | 0.332 | 0.105 | 0.332 | 0.465 |  | 0.912 |
| `mfcc45` | `rf` | 0.245 | 0.090 | 0.245 | 0.362 |  | 0.912 |
| `mfcc_sdc_mod_k7` | `rf` | 0.200 | 0.080 | 0.200 | 0.302 |  | 0.912 |

