---
title:  "Using Python's argparser"
categories: ["Python"]
tags: ["best practises"]
mathjax: fasle
---


In this post I am sharing best practises on using Python's `argparser`. This is common feedback I have given in pull request reviews over the past few years. 

Why would you care:
- Creating meaningful argument parsing is a skill. It can really make a difference on how useful your code is and how easily others can use it. 

There are at least two main common error patterns I observed: 
- How to pass booleans
- How to pass lists
Both are very reasonable things to want to do; each gets a small section below. 


## Booleans
Often we want to pass boolean arguments to control execution. A very frequent pattern is this one: 
```python
parser.add_argument("--create-report",
                    type=bool,        
                    default=True,
                    help="True if you want to write a report, False otherwise.")                            
```

This simply does not work. To convince yourself: 
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my_bool", type=bool)
cmd_line = ["--my_bool", "False"]
print(parser.parse_args(cmd_line))
# returns Namespace(my_bool=True)
```

How to do it [StackOverflow reference](https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/15008806#15008806):

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--my_bool',  action='store_true')

cmd_line = ["--my_bool" ]
print(parser.parse_args(cmd_line))
# returns Namespace(my_bool=True)
cmd_line = []
print(parser.parse_args(cmd_line))
# returns Namespace(my_bool=False)
```

## Lists
Another reasonable need is to want to pass lists. Example: 

```python
parser.add_argument("--my-categories",
                    type=list,        
                    default=['005', '00T'],
                    help="List of categories to filter data.")    
```

This simply does not work. To convince yourself: 
```python
parser = argparse.ArgumentParser()
parser.add_argument("--my-categories", type=list)
cmd_line = ["--my-categories", "['005', '500']"]
print(parser.parse_args(cmd_line))
# returns Namespace(my_categories=['[', "'", '0', '0', '5', "'", ',', ' ', "'", '5', '0', '0', "'", ']'])
```


or you may be tempted to do: 
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my-categories", type=str )
cmd_line = ["--my-categories", "['005', '500']"]
print(parser.parse_args(cmd_line))
# returns Namespace(my_categories="['005', '500']")
# then you go and apply ast.literal_eval
```

But this is not good separation of concerns. `argparser` can do argument validation for you for example. And you can achieve this out-of-the-box.


How to do it [StackOverflow reference](https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse/15753721#15753721)
```python
# this is test.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--my-categories", 
                    nargs='+', type=str) # type: refers to each elems type
args = parser.parse_args()
print(args.my_categories)
```
And here is how it works:
```python
python test.py --my-categories 005 006 007
# ['005', '006', '007'] 
```

