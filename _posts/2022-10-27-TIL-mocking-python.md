---
title:  "TIL: Using mocks and patching in Python efficiently"
categories: ["Python", "pytest"]
tags: ["Python", "testing"]
mathjax: false
---

In this post I am explaining things I learned when trying to run effective 
Python unit tests that required to mock some expensive resources. This is a very common need in 
practise when writing tests for machine learning code: 

> Load a model and use its predictions in some downstream tasks. 
> Repeat with another model with predictions of different output shape. And so on... 


Until now, I was "paying the price" to load the model to have its predictions in 
the tests without extra effort when coding the test. 
This is problematic in that test execution time increases fast and the 
CPU time is spent on loading/unloading models. Also, it is limiting because in environments
without internet access things will break. Even when there is internet access and a clean 
environment is created every time based on a Docker image for example, 
it is a waste of resources if you need to download the model say
from Huggingface Hub every time as part of a CI build. To fix suboptimal use of time and resources , 
I started  mocking and patching as now I feel much mre comfortable with the process.


What I had issues successfully grasping in earlier attempts was how to patch the correct class path 
for the functionality I wanted
and then how to correctly set the return values I needed. I was spending most of my time 
on trial and error or looking at various Stackoverflow posts trying to replicate.
I finally believe I understand better what I was missing. I will demonstrate 
in the example below the few principles/tricks I am currently using on an example.

Let's start by defining a simple class we will later test. For the moment it will simply 
load a sentence-transformer model and implement a method to get a minimal preprocessing 
and predictions from the model. In reality, such a class has more functionality to serve a 
downstream purpose, but it is sufficient for demonstration purposes. 

```python
# File fancy_ml_code.py
"""Some basic functionality we will later test"""
import logging
from sentence_transformers import SentenceTransformer


class FancyPredictor:
    """Loads and calls a tranformer model"""
    def __init__(self, model_name: str):
        """Toy init that just loads a model"""
        self.model = SentenceTransformer(model_name)

    def call(self, batch: list):
        """Loads self.model_name and tranforms a batch to embeddings"""
        logging.info(f'Batch size: {len(batch)}')
        batch = list(map(str.lower, batch))
        return self.model.encode(batch)
```

There is a constructore method that simply loads the sentence-transformer model and `call` that 
given a list `batch` lowercases each element and return the model's output. What is expensive in 
this case is loading the model. Transformer models are big and simply loading the model
requires several seconds. Things are worse if you need to get the model from the internet. This makes 
`self.model` an excellent 'resource' to mock. Let's start looking at few tests: 

```python
# File test_ml_code.py

import unittest
import numpy as np
from fancy_ml_code import FancyPredictor


class FancyPredictorTestCase(unittest.TestCase):

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_init(self, mocked_sentence_transformer):
        _ = FancyPredictor(self.model_name)
        self.assertEqual(mocked_sentence_transformer.call_count, 1)
        mocked_sentence_transformer.assert_called_with(self.model_name)
```
In this first test we patch `SentenceTransformer`: notice the 
path `'fancy_ml_code.SentenceTransformer'`. Simply trying to `@unittest.mock.patch('SentenceTransformer')`
would fail as we need to patch the `SentenceTranformer` used in `fancy_ml_code.py`. 
The test then checks that the class is called once and also 
checks the argument it is called with. Checking these is probably meaningful because it ensure that 
the expensive process (loading the model) will be done once in the code execution. 

Here is a second test that belongs to the same `FancyPredictorTestCase` class:
```python
    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_call(self, mocked_sentence_transformer):
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        with self.assertLogs(level='INFO') as cm:
            predictor.call(batch=batch)
        # Evaluate logs, both number and content
        self.assertEqual([log.msg for log in cm.records], ['Batch size: 2'])

        # Evaluate what was the `predict` arguments
        instance = mocked_sentence_transformer.return_value
        self.assertEqual(instance.encode.call_count, 1)
        instance.encode.assert_called_with(['first batch item','second batch item'])
        # Call `call` again and verify that the model is still initialized once
        predictor.call(batch=batch)
        self.assertEqual(mocked_sentence_transformer.call_count, 1)
        # But the `encode` twice
        self.assertEqual(instance.encode.call_count, 2)
```
The test here checks more things while patching again the expensive resource.
The test checks: 
### The emitted logs
This is useful if you populate a logline for instance) with `self.assertLogs`. 
There is a small caveat here: the test can only validate the actual log content. 
If one uses a log formatter, the formatting is lost when executing tests with pytest.

### The call of `encode`
The fact that the `encode` method of the `SentenceTransformer` model was called only once, *and* its arguments.
Accessing the arguments validates the text preprocessing for example. 
I had difficulty in grasping the semantics of this. 
But, essentially the trick is that to get access to the `encode` arguments and everything that is related to 
it (number of calls for example) you need to pass by `mocked_sentence_transformer.return_value`. 
This, because, `mocked_sentence_transformer` is the class mock and here we want to see what happend after 
we instantiated an object (the class return value is the object). From the pbject we can then access 
`encode` and validate its mock calls both their count and the arguments. 

Here is another test that shows how to mock the return values of encode, which can be used for other
more functional tests: 
```python
    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_functional(self, mocked_sentence_transformer):
        """Demonstrates of mocking return results for more functional tests"""
        expected_embeddings = np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]])
        mocked_sentence_transformer.return_value.encode.return_value = expected_embeddings
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        actual_embeddings = predictor.call(batch=batch)
        self.assertTrue((actual_embeddings == expected_embeddings).all())
```
In this test we do again the same trick: from mocking the class we access the mocked object (
`mocked_sentence_transformer.return_value`) and then we set the `return_value` of the `encode` method. 
This is very powerful. Here the code simply validates the behavior i.e., that we got 
the expected value but a lot of things can be built on top of this. 

This is the set of tricks I am now using to write small atomic tests that depend on much less resources.
The code effort is small, few lines to describe the behavior and the side effects of the mocks but the 
gains are impressive in terms of flexibility. For a comparison, a basic test without 
mocking that loads the actual model looks like:
```python
    def test_fancy_predictor_functional_without_mocking(self):
        """Demonstrates of mocking return results for more functional tests"""
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        actual_embeddings = predictor.call(batch=batch)
        self.assertTrue(actual_embeddings.shape == (2, 384))
```
For reference, this test alone takes ~8 seconds, whereas with mocking all other tests complete in
less than 0.01 second. With these timings, it is quite easy to imagine how 
with few tests requiring expensive models
we can end up waiting a lot for the tests. 

For reference here are the two file contents used in the examples of this post. You can run the tests 
and also see the timings with:
```bash
pytest --durations=0 -v  test_ml_code.py
```

```python
#fancy_ml_code.py
import logging
from sentence_transformers import SentenceTransformer


class FancyPredictor:
    """Loads and calls a tranformer model"""
    def __init__(self, model_name):
        """Toy init that just loads a model"""
        self.model = SentenceTransformer(model_name)

    def call(self, batch):
        """Loads self.model_name and tranforms a batch to embeddings"""
        logging.info(f'Batch size: {len(batch)}')
        batch = list(map(str.lower, batch))
        return self.model.encode(batch)
```

```python
#test_ml_code.py
import unittest
import numpy as np
from fancy_ml_code import FancyPredictor


class FancyPredictorTestCase(unittest.TestCase):

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_init(self, mocked_sentence_transformer):
        _ = FancyPredictor(self.model_name)
        self.assertEqual(mocked_sentence_transformer.call_count, 1)
        mocked_sentence_transformer.assert_called_with(self.model_name)

    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_call(self, mocked_sentence_transformer):
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        with self.assertLogs(level='INFO') as cm:
            predictor.call(batch=batch)
        # Evaluate logs, both number and content
        self.assertEqual([log.msg for log in cm.records], ['Batch size: 2'])

        # Evaluate what was the `predict` arguments
        instance = mocked_sentence_transformer.return_value
        self.assertEqual(instance.encode.call_count, 1)
        instance.encode.assert_called_with(['first batch item','second batch item'])
        # Call `call` again and verify that the model is still initialized once
        predictor.call(batch=batch)
        self.assertEqual(mocked_sentence_transformer.call_count, 1)
        # But the `encode` twice
        self.assertEqual(instance.encode.call_count, 2)

    @unittest.mock.patch('fancy_ml_code.SentenceTransformer')
    def test_fancy_predictor_functional(self, mocked_sentence_transformer):
        """Demonstrates of mocking return results for more functional tests"""
        expected_embeddings = np.array([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]])
        mocked_sentence_transformer.return_value.encode.return_value = expected_embeddings
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        actual_embeddings = predictor.call(batch=batch)
        self.assertTrue((actual_embeddings == expected_embeddings).all())

    def test_fancy_predictor_functional_without_mocking(self):
        """Demonstrates of mocking return results for more functional tests"""
        predictor = FancyPredictor(self.model_name)
        batch = ['First batch item', 'second batch ITEM']
        actual_embeddings = predictor.call(batch=batch)
        self.assertTrue(actual_embeddings.shape == (2, 384))

```
