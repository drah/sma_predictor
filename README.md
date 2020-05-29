# SMA_Predictor

## Usage

Put the data to this root directory, or use --data <path_to_data> to specify the path to the data.
### Train and Evaluation
`python3 main.py`

### Take the best checkpoint to evaluate on test
`python3 main.py --phase 0 --ckpt <path_to_ckpt>`

### See the predicted result
`python3 main.py --phase 2 --ckpt <path_to_ckpt>`
