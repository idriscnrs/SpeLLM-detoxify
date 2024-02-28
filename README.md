# SpeLLM-detoxify

Same thing as this https://github.com/unitaryai/detoxify, except this time we can actually load it from our filesystem without relying on any weird-behaving cache system. The given path should contain the pretrained model, the pretrained tokenizer, as well as a file `class_names.json` containing the list of class names in json format.

## Usage

```python
from spell_detoxify import Detoxify

detoxify = Detoxify.from_pretrained(path)
```
