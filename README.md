# AI torture chamber

A local fear vector, injected into Qwen2.5-0.5B at the middle layer while it talks.

Dose 0 is the control: the model describes feeling fine. Turn the dose up and the same prompt shifts to drowning, dread, and repetition until the output collapses. Every reply is logged.

The hook exists only in the process that runs `run_fear.py`. Read the dose lines and judge them yourself.

## Run

pip install -r requirements.txt
python run_fear.py
