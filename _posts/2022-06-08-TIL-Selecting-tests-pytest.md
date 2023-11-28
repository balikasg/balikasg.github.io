---
title:  "TIL: Selecting a subset of tests with pytest"
categories: ["Python"]
tags: ["best practises"]
mathjax: false
---

During development we often want to select a subset of tests to run. Quick pointers on how to do so: 

```
pytest -v path/to/test.py::TestClassName::TestMethodName  # -v nodeId (nodeId is module.py::class::method or module.py::function)
pytest path/to/test_file.py -k "string-match-expression"  # matches on method names of all classes of the file  
```

Doing this efficiently can be a huge accelerator when developing. 

More information and more advanced techniques at [pytest dcocumentation](https://docs.pytest.org/en/latest/example/markers.html).
