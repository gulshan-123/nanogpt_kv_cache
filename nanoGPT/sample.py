"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import numpy as np
import matplotlib.pyplot as plt
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = device.type # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

import time
# 1. Create a Long Prompt to make the prefill phase significant
# This repeats a sentence ~100 times to create a prompt of ~1000 tokens
long_prompt = '''ACT I. SCENE I.
London. The Palace.

Enter KING EDWARD, LORD HASTINGS, and the DUKE OF NORFOLK.

KING EDWARD:
So is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
But tell me, Hastings, for thou know'st the heart
Of this shifting populace better than I,
How stands the temper of the common man?
Do they rejoice in this our peaceful time,
Or do they murmur still of Henry's fall?

LORD HASTINGS:
My liege, the people are but water,
Taking the shape of the vessel that holds them.
While the wine is poured and the bread is broke,
They cry 'God Save King Edward' with full throats.
But let not your highness be deceived by noise;
Silence is the herald of true intent,
And in the shadows of the market square,
Some tongues still wag with tales of ancient rights.

KING EDWARD:
Ancient rights! 'Tis the ghost of treason
That haunts the hollow skulls of idle men.
Norfolk, what say you? Are our borders safe?
Hath the French King sent word regarding Calais?

DUKE OF NORFOLK:
No word, my lord, save for the silent wind
That blows cold from across the narrow sea.
But spies report a gathering of ships,
Not for trade, methinks, but for a darker purpose.
The Earl of Richmond, banished though he be,
Finds coin and comfort in the French court's lap.

KING EDWARD:
Richmond! That name is like a canker in my ear.
I thought him weeded from this garden plot,
Yet still he grows, watered by foreign hate.
We must be vigilant. Send out more scouts,
Double the guard upon the white-cliffed shore.
I will not lose what blood hath bought so dear.

Enter QUEEN ELIZABETH.

QUEEN ELIZABETH:
My lord, why do you furrow your brow so deep?
The feast awaits, and music fills the hall,
Yet here you stand, conspiring with the dark.

KING EDWARD:
Not conspiring, sweet Bess, but preparing.
A king's eyes must be open while others sleep.
But come, let us not spoil the hour with fear.
Give me thy hand. We shall attend the dance.

[Exeunt KING EDWARD and QUEEN ELIZABETH.]

LORD HASTINGS:
(Aside)
He speaks of fear, yet walks into the light,
Blind to the vipers coiling at his feet.
Norfolk, a word, before you join the throng.

DUKE OF NORFOLK:
Speak, Hastings. My ears are open.

LORD HASTINGS:
Think you the King sits firmly on this throne?
The roots are shallow, planted in fresh blood.
If Richmond comes, with French gold in his purse,
Will the army stand, or will they turn the tide?

DUKE OF NORFOLK:
Treason, Hastings? Is that what you whisper?
Take care, for walls have ears, and shadows eyes.
I am true to Edward, while he wears the crown.'''
start_ids = encode(long_prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
print(f"Input prompt shape: {x.shape}")

# 2. Run generation loop measuring TOTAL time
print("Warmup run (populating cache)...")

# Store times for plotting
total_times = []
warmup_prompt= '''SYSTEM_INIT_SEQUENCE: ALPHA_992
CURRENT_DATE: 2088-11-04
LOCATION: Lunar Base - Sector 7 (The "Dark Side" Observatory)
SECURITY_LEVEL: CLEARANCE_OMNI
LOG_START_MARKER: [BEGIN_TRANSMISSION]

--- SECTION 1: NARRATIVE CONTEXT ---

The hum of the atmospheric recyclers was the only sound in the command deck, a rhythmic thrumming that usually comforted Dr. Aris Thorne. Today, however, it sounded like a countdown. Aris looked at the monitor, his reflection ghostly against the cascading streams of telemetry data. The anomaly they had found in the Keplar quadrant wasn't just a glitch; it was a signal. A structured, mathematical sequence repeating every prime number second.

"Report status," Commander Vance barked, stepping through the pneumatic doors. His boots clanged heavily on the metal grating. Vance was a military man, all sharp angles and impatience, whereas Aris was soft edges and hesitation.

"It’s the signal, Commander," Aris said, rubbing his eyes. "It’s changed. It’s no longer broadcasting prime numbers. It’s broadcasting... architecture."

Vance stopped beside the console. "Architecture? You mean blueprints?"

"Not for a building," Aris corrected, tapping a sequence on the holographic keyboard. "For a syntax. It looks like code, but organic. If I didn't know better, I'd say the star system is trying to write a Python script."

Vance scoffed. "Stars don't write Python, Doctor. Get me the raw data logs. I want to see the energy fluctuations for the last 48 hours. And tell me about the coolant leak in Sector 4. Is that related?"

"The coolant leak is fixed," Aris lied. It wasn't fixed. The pressure valve was holding at 92%, but if the signal intensity increased, the magnetic containment field would destabilize. If that happened, the text generation module of the station's AI would begin to hallucinate.'''
with torch.no_grad():
    with ctx:
        for k in range(10): # Run 5 times
            
            # Start timer BEFORE calling generate
            t0 = time.time()
            
            # We only generate a few new tokens (e.g., 10) because we only care 
            # about the speed of processing the input prompt (x)
            y, _ = model.generate((torch.tensor(encode(warmup_prompt), dtype=torch.long, device=device)[None, ...]), max_new_tokens=10, temperature=temperature, top_k=top_k)
            
            # End timer
            t1 = time.time()
            
            duration = t1 - t0
            # total_times.append(duration)
            
            status = "MISS (Slow)" if k == 0 else "HIT  (Fast)"
            print(f"Run {k}: {duration:.4f}s - {status}")

with torch.no_grad():
    with ctx:
        for k in range(10): # Run 5 times
            
            # Start timer BEFORE calling generate
            t0 = time.time()
            
            # We only generate a few new tokens (e.g., 10) because we only care 
            # about the speed of processing the input prompt (x)
            y, _ = model.generate((torch.tensor(encode(long_prompt), dtype=torch.long, device=device)[None, ...]), max_new_tokens=1, temperature=temperature, top_k=top_k)
            
            # End timer
            t1 = time.time()
            
            duration = t1 - t0
            total_times.append(duration)
            
            status = "MISS (Slow)" if k == 0 else "HIT  (Fast)"
            print(f"Run {k}: {duration:.4f}s - {status}")

# 3. Simple Plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(total_times)), total_times, marker='o', linestyle='-', color='b')
plt.title("Impact of Shared Prefix Caching on Generation Time")
plt.xlabel("Iteration Number")
plt.ylabel("Total Execution Time (seconds)")
plt.xticks(range(len(total_times)))
plt.grid(True)

# Add text labels
plt.text(0, total_times[0], f" {total_times[0]:.2f}s (Cache Fill)", verticalalignment='bottom')
for i in range(1, len(total_times)):
    plt.text(i, total_times[i], f" {total_times[i]:.2f}s (Cache Hit)", verticalalignment='bottom')

plt.show()