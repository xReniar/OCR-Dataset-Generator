# Add new Generator
To add a new generator (suppose the OCR system is called `CUSTOM`):
- create a `custom.py` file in `./src/generator/custom.py`. If the generator is a pipeline of 2 ocr-tools create a `det_rec.py`
- add in `OCR_SYSTEMS` located in `__init__.py` the generator. (This associates the key to the `.py` file)
```py
from .custom import CUSTOMGenerator
from .det_rec import DETRECGenerator

OCY_SYSTEMS = {
    "custom": CUSTOMGenerator # add generator
    "det+rec": DETRECGenerator # add pipeline of 2 generators
}
```
- inside `custom.py` paste the code below:
```py
from .generator import Generator
from ..dataloader import Dataloader


class CUSTOMGenerator(Generator):
    def __init__(self,test_name: str,datasets: list, lang, transforms) -> None:
        super().__init__(test_name,datasets,lang,transforms)

    def _generate(self,dataloader: Dataloader,task: str, process) -> None:
        pass

    def _det(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass

    def _rec(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass
```