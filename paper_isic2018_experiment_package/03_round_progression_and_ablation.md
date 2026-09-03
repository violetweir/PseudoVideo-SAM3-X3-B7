# Round Progression And Ablation

## Student Progression

| Student | Iteration | Val Dice | Test Dice @256 | Note |
|---|---|---|---|---|
| S2 | 12000 | 0.850589 | 0.848341 | complete |
| S3 | 26600 | 0.852072 | 0.854872 | complete |
| X3_best | 22400 | 0.869182 | 0.859377 | complete |
| X3_final | 40000 | 0.864878 | 0.864459 | complete |
| X4_best | 24000 | 0.870790 | 0.870414 | complete |
| X4_final | 40000 | 0.867375 |  | final checkpoint test not separately evaluated |

## B7 Selector Summaries

| Setting | Selected GT Dice | Oracle GT Dice | Coverage |
|---|---|---|---|
| round1_base_x3_best | 0.761549 | 0.788188 | 1.0 |
| round2a_epoch27_x3_best | 0.861104 | 0.878812 | 1.0 |
| round2a_epoch27_x4_best | 0.860339 | 0.878812 | 1.0 |

The B7 selector values are useful for analyzing selector quality and potential teacher/bridge choices, but should not be written as standalone student-mask Dice.
