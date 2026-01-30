# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# persistent self (optional)
self_state_enabled = False
self_state_dim = 32
self_stats_dim = 5
self_state_decay = 0.2
self_update_type = 'mlp'
self_state_init = 'checkpoint'
self_state_init_scale = 1.0
self_state_init_norm_ref = ''
self_memory_enabled = False
self_memory_decay = 0.01
self_memory_scale = 1.0
self_memory_mode = 'buffer'
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

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
