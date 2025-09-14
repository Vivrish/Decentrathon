# Application for a hackathon - detects car state

# Run

1. install docker
2. docker compose up

# Development

Backend:

1. Create a venv for python, activate it
2. pip install -r requirements.txt

# P.S after the deadline
We realized some links were missing in the initial submission.

Here is the link to our program that adds VLM comments to YOLO results:
https://drive.google.com/file/d/1xh73ccqEXrqpWPtqG7Cv33EGm4txCua2/view?usp=sharing
Since the weights of VLM are too heavy to be placed on a server, as well as stored on a git, we decided to keep it as a folder in drive.
Unzip it in your workspace.

Then follow these instructions:
1. paste an image into the folder called images
2. run 'pip install -r requirements.txt'
3. run 'python program_VLM.py test_damaged.jpg --models-file models.txt --classes-file classes.txt --dirt-classes "dirt" --use_vlm true'

Demo:
https://drive.google.com/file/d/1vSgzARPrJfFY-rS_VI_Zc6ntiDalvJ_e/view?usp=sharing
