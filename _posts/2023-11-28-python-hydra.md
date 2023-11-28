---
title: "Using `hydra` for config management in Python"
categories: ["Python"]
tags: ["best practises", "config"]
mathjax: false
---

Often in my day-to-day I deal with configs that control either experiments or initializations of 
python classes (say the hyper-parameter of an ML model). 
The use-case is as-follows: there is always a `config.yaml` that the python code 
reads. This works well in the first iterations but once we start making 
manual changes to the `config.yaml` to replace default it which soon becomes a nightmare.

The next improvement is to manually configure an argparser to replace such config values 
passed from the terminal before class initialization and add some logging to keep track of it. 
But this is also limiting as it requires extra code and logic and does not generalise across projects.

Today, I discovered [`hydra`](https://github.com/facebookresearch/hydra) which does much more on this front and here I detail it. 

## The manual implementation based on `argparser` 
Let's assume we work on an NLP project and we will deal with stopwords. The user should define their laguage. 
There is a config file in the project `config.yaml`:
```bash
preprocessing:
  stopwords: italian
```

To implement argparsing updates, we would start with a `Config` class that simply reads from a yaml and implements the logic.
Here is a skeleton: 
```python3
# This is the main.py code
from typing import Dict, List, Union
from pathlib import Path
from functools import reduce
import operator, argparse, yaml, json


class Config:
    """Toy class to represent a config"""

    def __init__(self, path: Union[Path, str]):
        self.config: Dict = self.load(path)

    @staticmethod
    def load(path: Union[Path, str]):
        """Loads a config from a yaml file"""
        with open(path, 'r') as my_yaml:
            config = yaml.safe_load(my_yaml)
        return config

    def set(self, keys: List[str], value: str):
        """Sets the config value to `value` in a nested dictionary
        following the keys in `keys`.
        Ref. https://stackoverflow.com/a/14692747"""
        self.getFromDict(self.config, keys[:-1])[keys[-1]] = value

    @staticmethod
    def getFromDict(dataDict: Dict, key_list: List[str]):
        """Given a dictionary `dataDict` traverses it using the arguments in
        `key_list`.
        Ref. https://stackoverflow.com/a/14692747"""
        return reduce(operator.getitem, key_list, dataDict)

    def __str__(self):
        """Print as a json with indent to be readable"""
        return json.dumps(self.config, indent=2)


def get_parser():
    """Returns a toy argparser"""
    parser = argparse.ArgumentParser(description='Toy Parser')
    parser.add_argument('--config', required=True)
    return parser


def overwrite_args(config_path: Union[Path, str], unknown_arguments: List):
    """Allows to handle extra args in the form of config.stopwords english"""
    config = Config(config_path)
    # TBD: sanity checks, logging, type-casting..
    for arg_key, arg_value in zip(unknown_arguments[0::2], unknown_arguments[1::2]):
        # Ignore `config` in the argument
        key_list = arg_key.split(".")[1:]
        config.set(key_list, arg_value)
    return config


if __name__ == "__main__":
    known_args, unknown_arguments = get_parser().parse_known_args()
    # Load config and prepare if overrides are requested
    input_config = overwrite_args(known_args.config, unknown_arguments)
    print(input_config)
    # Initialize classes ...
```

The code above is mostly self-explanatory. There are two main tricks: 
- The argparser keeps track of unexpected arguments, dubbed `unknown_arguments`. 
- The code expects `unknown_arguments` to be in pairs, the first item is the path to the value to be updated, the second the value. 
- Example: appending `--config.stopwords english` or `--config.stopwords french` should set the value of stopwords in the config to `english` or `french` respectively. 

For brevity in the implementation I did not code all the sanity checks (eg, pairs of arguments), 
type-casting eg, to integer or floats, logging, checking if the path is actually in the config or defines a new value.. 
This works as follows: 
```bash
python main.py --config config.yaml 
```
returns as expected the config values: 
```bash
{
  "preprocessing": {
    "stopwords": "italian"
  }
}
```
but passing:
```bash 
python main.py --config config.yaml --config.preprocessing.stopwords english
```
does the override: 
```shell
{
  "preprocessing": {
    "stopwords": "english"
  }
}
```

Now recall my comment: 
> I did not code all the sanity checks (eg, pairs of arguments), 
type-casting eg, to integer or floats, logging, checking if the path is actually in the config or defines a new value..

Well, `hydra` does all these and even more with minimal code, something that I liked.

## Reimplementing the same with hydra
Assuming again that there is a config file in the project `config.yaml`:
```bash
preprocessing:
  stopwords: italian
```
We install the library: 
```bash
pip install hydra-core --upgrade
```
this is how we do the same with it ([tutorial](https://hydra.cc/docs/intro/)): 
```python 
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./", config_name="config")
def get_config(cfg: DictConfig) -> DictConfig:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    get_config()
```
Now running:
```
python main2.py
```
returns the defaults: 
```shell
preprocessing:
  stopwords: italian
```
But:
```shell
python main2.py preprocessing.stopwords=english
```
results in: 
```shell
preprocessing:
  stopwords: english
```
which is the intended behavior. 

There is a catch in this where I spent quite some time: 
if in the `get_config` function we try to return the config as we do with argparser and follow 
this logic we will end-up returning `None`. I spent a lot of time trying to figure this out and 
it is the expected behavior. The framework expects the function that is tagged 
with `hydra.main` to be the code entry-point. This is used with 
[multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) (I don't cover it here). 
More details on this [github issue](https://github.com/facebookresearch/hydra/issues/332). 

I find this very neat, and there is a ton more stuff I did not cover here to be explored. 
But it definitely saves a lot of coding on the expense on onboarding to the framework of course. 