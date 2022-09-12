# Wenet run on azure itp
Copy from wenet project. Save data and model to azure blob.
The run bash script is in script

# Start

## Data Preparation
We extract data preparation from wenet project for common data prepare
We need follow files:
- wav.scp
- text
Data prepare flow path:
1. cmvn extract
2. dictionary build
3. create json file
```bash
cd yaml/wenet
amlt run data_prepare.yaml
```

## Train
After data preparation we can run the aishell recipe
```bash
cd yaml/wenet
amlt run train.yaml
```

## Test
Test model
```bash
cd yaml/wenet
amlt run test.yaml
```
