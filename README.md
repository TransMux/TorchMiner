This Project is Forked From [MineTorch](https://github.com/louis-she/minetorch).

Published on [pypi](https://pypi.org/project/torchminer/)

Packaged Using [Poetry](https://python-poetry.org/)

# Description
TorchMiner is designed to automatic process the training ,evaluating and testing process for PyTorch DeepLearning,with a simple API.

You can access all Functions of MineTorch simply use `Miner`.

## Project ToDo
 [!] compatible with paddlepaddle

 [!!!] Test Cases
 
 [!] Add A thread to accept CLI input when training
 
 [!] Abstract Miner process, for easier patches
 
 [] Abstract Plugin Manager

 [] Move ***Drawer*** Operations Outside of Miner as a Plugin
 
 [] A Plugin that can record every output of network for future analysis
 
 [] Add Plugin Able And Disable Stat
 
 [] Move Miner Options to yaml File, Add Config Class
 
 Now Plugins only supports output functions, they can't modify or change the data of the Miner class.Any Ideas? I am glad to know.
 
 [] Write about my design concept
 
 Critical 
 
 [] Deal About the input size problem, such as Batch-first...