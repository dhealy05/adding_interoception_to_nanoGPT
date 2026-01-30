"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from regimes import RegimeManager, build_legacy_bewilderment_regimes, load_regimes_from_path
from interventions import (
    apply_interventions,
    apply_self_interventions,
    apply_stats_interventions,
    compute_lr_multiplier,
)
from reaction import ReactionRecorder

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# persistent self (optional)
self_state_enabled = False
self_state_dim = 32
self_stats_dim = 5
self_state_decay = 0.2
self_update_type = 'mlp' # 'mlp', 'gru', or 'linear'
self_state_init = 'checkpoint' # 'checkpoint', 'zero', or 'random'
self_state_init_scale = 1.0 # scales random init norm; uses checkpoint norm if available
self_state_init_norm_ref = '' # optional ckpt path to match norm against
self_memory_enabled = False
self_memory_decay = 0.01
self_memory_scale = 1.0
self_memory_mode = 'buffer'  # 'ema' or 'buffer'
self_memory_buffer_len = 64
self_memory_buffer_stride = 10
self_memory_feat_dim = 8
self_memory_clamp_k = 1.0
self_memory_use_projection = False
self_state_reset_at = -1
self_state_log_interval = 100
self_state_effect_interval = 0
self_state_dump_interval = 0
self_state_dump_dir = ''
self_state_dump_max = 0
# reaction logging (optional)
reaction_log = False
reaction_log_path = ''
reaction_log_interval = 1
reaction_log_flush_interval = 100
# regime config (optional)
regime_config_path = ''
# bewilderment patch (optional)
bewilderment_patch = False
bewilderment_patch_start = 2000
bewilderment_patch_end = 2200
bewilderment_patch_prob = 0.5
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(__file__), 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    self_state_enabled=self_state_enabled,
    self_state_dim=self_state_dim,
    self_stats_dim=self_stats_dim,
    self_state_decay=self_state_decay,
    self_update_type=self_update_type,
    self_memory_enabled=self_memory_enabled,
    self_memory_decay=self_memory_decay,
    self_memory_scale=self_memory_scale,
    self_memory_feat_dim=self_memory_feat_dim,
    self_memory_clamp_k=self_memory_clamp_k,
    self_memory_use_projection=self_memory_use_projection,
) # start with model_args from command line
checkpoint_self_state = None
checkpoint_stats_prev = None
checkpoint_self_memory_ema = None
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    ckpt_config = checkpoint.get('config', {})
    if 'self_memory_mode' in ckpt_config:
        if self_memory_mode != ckpt_config['self_memory_mode']:
            print(f"overriding self_memory_mode from checkpoint: {ckpt_config['self_memory_mode']}")
        self_memory_mode = ckpt_config['self_memory_mode']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # keep self-state settings consistent with the checkpoint, if present
    if 'self_state_enabled' in checkpoint_model_args:
        for k in [
            'self_state_enabled',
            'self_state_dim',
            'self_stats_dim',
            'self_state_decay',
            'self_update_type',
            'self_memory_enabled',
            'self_memory_decay',
            'self_memory_scale',
            'self_memory_feat_dim',
            'self_memory_clamp_k',
            'self_memory_use_projection',
        ]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
    else:
        model_args['self_state_enabled'] = False
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    checkpoint_self_state = checkpoint.get('self_state', None)
    checkpoint_stats_prev = checkpoint.get('self_stats_prev', None)
    checkpoint_self_memory_ema = checkpoint.get('self_memory_ema', None)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
self_state_enabled = model.config.self_state_enabled
self_state_dim = model.config.self_state_dim
self_stats_dim = model.config.self_stats_dim
self_state_decay = model.config.self_state_decay
self_update_type = model.config.self_update_type
self_memory_enabled = model.config.self_memory_enabled
self_memory_decay = model.config.self_memory_decay
self_memory_scale = model.config.self_memory_scale
self_memory_feat_dim = model.config.self_memory_feat_dim
self_memory_clamp_k = model.config.self_memory_clamp_k
self_memory_use_projection = model.config.self_memory_use_projection
config['self_memory_enabled'] = model.config.self_memory_enabled
config['self_memory_decay'] = model.config.self_memory_decay
config['self_memory_scale'] = model.config.self_memory_scale
config['self_memory_feat_dim'] = model.config.self_memory_feat_dim
config['self_memory_clamp_k'] = model.config.self_memory_clamp_k
config['self_memory_use_projection'] = model.config.self_memory_use_projection
config['self_memory_mode'] = self_memory_mode
config['self_state_enabled'] = self_state_enabled
config['self_state_dim'] = self_state_dim
config['self_stats_dim'] = self_stats_dim
config['self_state_decay'] = self_state_decay
config['self_update_type'] = self_update_type

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# initialize persistent self state (if enabled)
def _load_self_state_norm(path):
    ckpt = torch.load(path, map_location='cpu')
    if 'self_state' not in ckpt:
        raise ValueError(f"norm ref checkpoint missing self_state: {path}")
    return ckpt['self_state'].float().norm().item()

if self_state_enabled:
    init_mode = str(self_state_init).lower()
    if init_mode == 'checkpoint':
        if checkpoint_self_state is not None:
            self_state = checkpoint_self_state.to(device)
        else:
            self_state = torch.zeros(self_state_dim, device=device)
    elif init_mode == 'zero':
        self_state = torch.zeros(self_state_dim, device=device)
    elif init_mode == 'random':
        self_state = torch.randn(self_state_dim, device=device)
        if self_state_init_norm_ref:
            ref_norm = _load_self_state_norm(self_state_init_norm_ref)
        elif checkpoint_self_state is not None:
            ref_norm = checkpoint_self_state.float().norm().item()
        else:
            ref_norm = 1.0
        target_norm = float(self_state_init_scale) * float(ref_norm)
        if target_norm <= 0.0:
            self_state.zero_()
        else:
            cur_norm = self_state.norm().item()
            if cur_norm == 0.0:
                self_state.zero_()
            else:
                self_state = self_state * (target_norm / cur_norm)
    else:
        raise ValueError(f"unknown self_state_init: {self_state_init}")
    if checkpoint_stats_prev is not None:
        stats_prev = checkpoint_stats_prev.to(device)
    else:
        stats_prev = torch.zeros(self_stats_dim, device=device)
    self_state_ema = self_state.detach().clone()
    if self_memory_enabled and self_memory_mode == 'ema':
        if checkpoint_self_memory_ema is not None:
            self_memory_ema = checkpoint_self_memory_ema.to(device)
        else:
            self_memory_ema = self_state.detach().clone()
    else:
        self_memory_ema = None
    if self_memory_enabled and self_memory_mode == 'buffer':
        if self_memory_buffer_len <= 0:
            raise ValueError("self_memory_buffer_len must be > 0 when self_memory_mode='buffer'")
        self_memory_buffer = torch.zeros((self_memory_buffer_len, self_state_dim), device=device)
        self_memory_buffer_idx = 0
        self_memory_buffer_count = 0
    else:
        self_memory_buffer = None
        self_memory_buffer_idx = 0
        self_memory_buffer_count = 0
    if not self_state_dump_dir:
        self_state_dump_dir = out_dir
    if master_process and self_state_dump_interval > 0:
        os.makedirs(self_state_dump_dir, exist_ok=True)
    self_state_dump_count = 0
else:
    self_state = None
    stats_prev = None
    self_state_ema = None
    self_memory_ema = None
    self_state_dump_count = 0
    self_memory_buffer = None
    self_memory_buffer_idx = 0
    self_memory_buffer_count = 0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, self_state=self_state)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def build_self_stats(logits, loss, grad_norm, step, max_steps, device):
    logits_slice = logits.detach().float()[:, -1, :]
    log_probs = F.log_softmax(logits_slice, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    top1_conf = probs.max(dim=-1).values.mean()
    loss_scalar = loss.detach().float().item() * gradient_accumulation_steps
    grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0
    step_frac = step / max_steps
    return torch.tensor(
        [loss_scalar, grad_norm_val, step_frac, entropy.item(), top1_conf.item()],
        device=device,
    )

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

if self_state_enabled and self_stats_dim != 5:
    raise ValueError("self_stats_dim must be 5 for the built-in stats vector")
if self_state_dump_interval < 0:
    raise ValueError("self_state_dump_interval must be >= 0")
if self_memory_enabled and (self_memory_decay < 0.0 or self_memory_decay > 1.0):
    raise ValueError("self_memory_decay must be in [0, 1]")
if self_memory_mode not in ("ema", "buffer"):
    raise ValueError("self_memory_mode must be 'ema' or 'buffer'")
if self_memory_buffer_len < 0:
    raise ValueError("self_memory_buffer_len must be >= 0")
if self_memory_buffer_stride <= 0:
    raise ValueError("self_memory_buffer_stride must be > 0")

if regime_config_path:
    regimes = load_regimes_from_path(regime_config_path)
else:
    regimes = build_legacy_bewilderment_regimes(
        bewilderment_patch,
        bewilderment_patch_start,
        bewilderment_patch_end,
        bewilderment_patch_prob,
    )
regime_manager = RegimeManager(regimes)
reaction_recorder = None
if reaction_log and master_process:
    log_path = reaction_log_path or os.path.join(out_dir, "reaction_log.jsonl")
    reaction_recorder = ReactionRecorder(log_path, flush_interval=reaction_log_flush_interval)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
prev_self_state = None
last_state_effect = None
last_update_mlp_norm = None
last_update_mem_norm = None
last_update_mem_clamped_norm = None
last_update_cos = None
last_update_gate = None
while True:

    # determine and set the learning rate for this iteration
    active_regimes = regime_manager.active(iter_num)
    lr = get_lr(iter_num) if decay_lr else learning_rate
    lr *= compute_lr_multiplier(active_regimes)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if self_state_enabled:
                    checkpoint['self_state'] = self_state.detach().cpu()
                    checkpoint['self_stats_prev'] = stats_prev.detach().cpu()
                    if self_memory_enabled and self_memory_mode == 'ema' and self_memory_ema is not None:
                        checkpoint['self_memory_ema'] = self_memory_ema.detach().cpu()
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    if self_state_enabled:
        if self_state_reset_at >= 0 and iter_num == self_state_reset_at:
            self_state.zero_()
            stats_prev.zero_()
            self_state_ema = self_state.detach().clone()
            if self_memory_ema is not None:
                self_memory_ema = self_state.detach().clone()
            if self_memory_buffer is not None:
                self_memory_buffer.zero_()
                self_memory_buffer_idx = 0
                self_memory_buffer_count = 0
        prev_self_state = self_state.detach()
        stats_for_update = stats_prev
        if active_regimes:
            stats_for_update = apply_stats_interventions(stats_for_update, active_regimes)
        memory_delta = None
        if self_memory_enabled:
            if self_memory_mode == 'ema':
                if self_memory_ema is None:
                    self_memory_ema = self_state.detach().clone()
                memory_delta = self_memory_ema - self_state.detach()
            elif self_memory_mode == 'buffer':
                if self_memory_buffer_count > 0 and self_memory_buffer is not None:
                    if self_memory_buffer_count < self_memory_buffer_len:
                        buf = self_memory_buffer[:self_memory_buffer_count]
                    else:
                        buf = self_memory_buffer
                    q = self_state.detach()
                    scores = torch.mv(buf, q) / math.sqrt(self_state_dim)
                    weights = torch.softmax(scores, dim=0)
                    context = (weights.unsqueeze(1) * buf).sum(dim=0)
                    memory_delta = context - self_state.detach()
                else:
                    memory_delta = None
            else:
                raise ValueError(f"unknown self_memory_mode: {self_memory_mode}")
        update_raw, update_mem, update_mem_clamped, update_gate = raw_model.self_state_controller.compute_updates(
            self_state.detach(),
            stats_for_update,
            memory_delta=memory_delta,
            memory_scale=self_memory_scale,
        )
        self_state = raw_model.self_state_controller.update_state(
            self_state.detach(),
            stats_for_update,
            memory_delta=memory_delta,
            memory_scale=self_memory_scale,
            precomputed=(update_raw, update_mem, update_mem_clamped, update_gate),
        )
        with torch.no_grad():
            raw_norm = update_raw.detach().norm().item()
            mem_norm = update_mem.detach().norm().item()
            mem_clamped_norm = update_mem_clamped.detach().norm().item()
            denom = raw_norm * mem_norm
            if denom > 1e-12:
                update_cos = F.cosine_similarity(update_raw.detach(), update_mem.detach(), dim=0).item()
            else:
                update_cos = 0.0
            gate_val = update_gate.detach().view(-1)[0].item()
        last_update_mlp_norm = raw_norm
        last_update_mem_norm = mem_norm
        last_update_mem_clamped_norm = mem_clamped_norm
        last_update_cos = update_cos
        last_update_gate = gate_val
        if active_regimes:
            self_state = apply_self_interventions(self_state, active_regimes)
        if self_memory_enabled:
            if self_memory_mode == 'ema':
                self_memory_ema = (1.0 - self_memory_decay) * self_memory_ema + self_memory_decay * self_state.detach()
            elif self_memory_mode == 'buffer':
                if self_memory_buffer is not None and (iter_num % self_memory_buffer_stride == 0):
                    self_memory_buffer[self_memory_buffer_idx].copy_(self_state.detach())
                    self_memory_buffer_idx = (self_memory_buffer_idx + 1) % self_memory_buffer_len
                    self_memory_buffer_count = min(self_memory_buffer_len, self_memory_buffer_count + 1)
        with torch.no_grad():
            self_state_ema = 0.99 * self_state_ema + 0.01 * self_state.detach()

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    loss_accum = None
    stats_entropy_sum = None
    stats_top1_sum = None
    stats_count = 0
    track_stats = self_state_enabled or reaction_recorder is not None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            X_step, Y_step = X, Y
            if active_regimes:
                X_step, Y_step = apply_interventions(X_step, Y_step, active_regimes, model.config)
            logits, loss = model(X_step, Y_step, self_state=self_state)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        loss_accum = loss.detach() if loss_accum is None else (loss_accum + loss.detach())
        if track_stats:
            with torch.no_grad():
                logits_slice = logits.detach().float()[:, -1, :]
                log_probs = F.log_softmax(logits_slice, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                top1 = probs.max(dim=-1).values.mean()
                stats_entropy_sum = entropy.detach() if stats_entropy_sum is None else (stats_entropy_sum + entropy.detach())
                stats_top1_sum = top1.detach() if stats_top1_sum is None else (stats_top1_sum + top1.detach())
                stats_count += 1
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = 0.0
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    loss_mean = loss_accum.item() if loss_accum is not None else 0.0
    stats_current = None
    if track_stats:
        if stats_count > 0 and stats_entropy_sum is not None and stats_top1_sum is not None:
            entropy_mean = (stats_entropy_sum / stats_count).item()
            top1_mean = (stats_top1_sum / stats_count).item()
        else:
            entropy_mean = 0.0
            top1_mean = 0.0
        grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0
        step_frac = iter_num / max_iters
        stats_current = torch.tensor(
            [loss_mean, grad_norm_val, step_frac, entropy_mean, top1_mean],
            device=device,
        )
    if self_state_enabled:
        stats_prev = stats_current

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # loss_mean is averaged across micro-steps when using gradient accumulation
        lossf = loss_mean
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if self_state_enabled and (iter_num % self_state_log_interval == 0):
            with torch.no_grad():
                self_norm = self_state.norm().item()
                if prev_self_state is None:
                    self_delta = 0.0
                else:
                    self_delta = (self_state - prev_self_state).norm().item()
                drift = 1.0 - F.cosine_similarity(self_state, self_state_ema, dim=0).item()
            print(f"self_norm={self_norm:.3f}, self_delta={self_delta:.3f}, self_drift={drift:.3f}")
        if self_state_enabled and self_state_effect_interval > 0 and (iter_num % self_state_effect_interval == 0):
            with torch.no_grad():
                test_len = min(4, model.config.vocab_size)
                test_input = torch.arange(test_len, device=device).unsqueeze(0)
                out_with_state, _ = model(test_input, self_state=self_state)
                out_zero_state, _ = model(test_input, self_state=torch.zeros_like(self_state))
                last_state_effect = (out_with_state - out_zero_state).abs().mean().item()
            print(f"state_effect={last_state_effect:.6f}")

    if self_state_enabled and master_process and self_state_dump_interval > 0:
        if iter_num % self_state_dump_interval == 0:
            if self_state_dump_max == 0 or self_state_dump_count < self_state_dump_max:
                dump_path = os.path.join(self_state_dump_dir, f"self_state_{iter_num}.pt")
                torch.save(self_state.detach().cpu(), dump_path)
                if stats_prev is not None:
                    stats_path = os.path.join(self_state_dump_dir, f"self_stats_{iter_num}.pt")
                    torch.save(stats_prev.detach().cpu(), stats_path)
                self_state_dump_count += 1
    if reaction_recorder is not None and (iter_num % reaction_log_interval == 0):
        metrics = {}
        metrics["loss"] = loss_mean
        metrics["lr"] = lr
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        metrics["step_frac"] = iter_num / max_iters
        if stats_current is not None:
            stats_cpu = stats_current.detach().float().cpu().tolist()
            if len(stats_cpu) >= 5:
                metrics["stats_loss"] = stats_cpu[0]
                metrics["stats_grad_norm"] = stats_cpu[1]
                metrics["stats_step_frac"] = stats_cpu[2]
                metrics["stats_entropy"] = stats_cpu[3]
                metrics["stats_top1_conf"] = stats_cpu[4]
        if self_state_enabled:
            with torch.no_grad():
                metrics["self_norm"] = self_state.norm().item()
                if prev_self_state is None:
                    metrics["self_delta"] = 0.0
                else:
                    metrics["self_delta"] = (self_state - prev_self_state).norm().item()
                metrics["self_drift"] = 1.0 - F.cosine_similarity(self_state, self_state_ema, dim=0).item()
        if last_state_effect is not None:
            metrics["state_effect"] = last_state_effect
        if last_update_mlp_norm is not None:
            metrics["self_update_mlp_norm"] = last_update_mlp_norm
            metrics["self_update_mem_norm"] = last_update_mem_norm
            metrics["self_update_mem_clamped_norm"] = last_update_mem_clamped_norm
            metrics["self_update_cos"] = last_update_cos
            metrics["self_update_gate"] = last_update_gate
        reaction_recorder.record(iter_num, active_regimes, metrics)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if reaction_recorder is not None:
    reaction_recorder.close()

if ddp:
    destroy_process_group()
