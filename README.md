# Wenet run on azure itp
Copy from wenet project.

# Start

## Data Preparation
We extract data preparation from wenet project for common data prepare
We need follow files:
- raw_wav
- text
Data prepare flow path:
1. cmvn extract
2. dictionary build
3. create json file

## Train
After data preparation we can run the aishell recipe
```bash
amlt run aishell.yaml
```


