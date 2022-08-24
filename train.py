# coding: utf-8
import argparse
import time
import math
import os
import sys
import itertools

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from utils import move_to_cuda, chunk_inputs

from data_utils import get_lm_corpus
from datetime import datetime
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    print(f'device = {device}')
    print(f'rank = {args.rank} || local_rank = {args.local_rank}')

    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        print(f'init_method = {init_method}')
        dist.init_process_group(backend=args.distributed_backend,
                                world_size=args.world_size,
                                rank=args.rank,
                                init_method=init_method)
        dist.all_reduce(torch.zeros(1).cuda())
        suppress_output(args.rank == 0)


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data',
                    type=str,
                    default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset',
                    type=str,
                    default='wt103',
                    choices=[
                        'wt103',
                        'lm1b',
                        'enwik8',
                        'text8',
                        'chinese-novel',
                        'fine-tune-kd',
                        'kd_continue_with_token',
                        'kd_continueToken_with_namefix',
                        'person_predict',
                        'summary_score',
                    ],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12, help='number of total layers')
parser.add_argument('--n_head', type=int, default=10, help='number of heads')
parser.add_argument('--d_head', type=int, default=50, help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1, help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500, help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000, help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0, help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0, help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str, help='parameter initializer to use.')
parser.add_argument('--init_range',
                    type=float,
                    default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range',
                    type=float,
                    default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std',
                    type=float,
                    default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std',
                    type=float,
                    default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim',
                    default='adam',
                    type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr',
                    type=float,
                    default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0, help='momentum for sgd')
parser.add_argument('--scheduler',
                    default='cosine',
                    type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')
parser.add_argument('--decay_rate',
                    type=float,
                    default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min',
                    type=float,
                    default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--clip_nonemb',
                    action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60, help='batch size')
parser.add_argument('--not_namepostfix',
                    action='store_true',
                    help='use name token but without namepostfix')
parser.add_argument('--valid-batch-size', type=int, default=60, help='batch size')
parser.add_argument('--batch_chunk',
                    type=int,
                    default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70, help='number of tokens to predict')
parser.add_argument('--eval_tgt_len',
                    type=int,
                    default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0, help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0, help='length of the retained previous heads')
parser.add_argument('--code_type', type=int, default=5, help='which version token type is used')
parser.add_argument('--not_tied',
                    action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--adaptive', action='store_true', help='use adaptive softmax')
parser.add_argument('--div_val',
                    type=int,
                    default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm',
                    action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument(
    '--add_lnorm',
    action='store_true',
    help='apply LayerNorm to the input instead of the output after the final output')
parser.add_argument('--varlen', action='store_true', help='use variable length')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200, help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000, help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str, help='experiment directory.')
parser.add_argument('--restart',
                    action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='', help='restart dir')
parser.add_argument('--debug',
                    action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length',
                    action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type',
                    type=int,
                    default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len',
                    type=int,
                    default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min',
                    type=float,
                    default=5e-5,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1, help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1, help='max eval steps')
parser.add_argument('--sample_softmax',
                    type=int,
                    default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0, help='patience')
parser.add_argument('--finetune_v2', action='store_true', help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true', help='finetune v3')
parser.add_argument('--fp16',
                    action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--autocast', action='store_true', help='Run in pytorch autocast mode.')
parser.add_argument('--optim_level',
                    type=str,
                    default='O1',
                    help='use optim level for amp in fp16 trainning')
parser.add_argument("--untie_r",
                    action='store_true',
                    help="untie r_w_bias and r_r_bias and r_s_bias")
parser.add_argument("--do_valid",
                    action='store_true',
                    default=False,
                    help="")
parser.add_argument("--with_token",
                    action='store_true',
                    help="with or without token_type_embedding")
parser.add_argument("--output_attention",
                    action='store_true',
                    help="with or without token_type_embedding")
###
parser.add_argument("--vocab-file", type=str, default="", help="vocabulary file of BertTokenizer")
parser.add_argument("--workers", type=int, default=6, help='number of tokenizer')
parser.add_argument("--save-model-name",
                    type=str,
                    default="",
                    help="name of directory in save model checkpoints")
parser.add_argument("--checkpoint_name",
                    type=str,
                    default="model.pt",
                    help="best checkpoint file name")
parser.add_argument('--skip-epoch',
                    type=int,
                    default=0,
                    help='Number of skip-epoch when resume training')
parser.add_argument('--skip-step',
                    type=int,
                    default=0,
                    help='Number of skip-step when resume training')
parser.add_argument('--distributed-backend',
                    default='nccl',
                    help='which backend to use for distributed training. One of [gloo, nccl]')
parser.add_argument('--local_rank',
                    type=int,
                    default=None,
                    help='local rank passed from distributed launcher')
parser.add_argument('--save-anyway', action='store_true', help='save checkpoint anyways')
args = parser.parse_args()

# ddp
args.cuda = torch.cuda.is_available()
args.rank = int(os.getenv('RANK', '0'))
args.world_size = int(os.getenv("WORLD_SIZE", '1'))
if args.world_size == 1:
    import torch.distributed as dist#wab
    dist.init_process_group('gloo', init_method='tcp://127.0.0.1:8929', rank=0, world_size=1)#wab
if args.world_size > 1:
    initialize_distributed(args)
set_random_seed(args.seed)

args.tied = not args.not_tied
args.namepostfix = not args.not_namepostfix

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
# assert args.batch_size % args.batch_chunk == 0

args.work_dir = os.path.join(args.work_dir, "{}_{}".format(args.save_model_name,
                                                           time.strftime('%Y%m%d')))
logging = create_exp_dir(args.work_dir,
                         scripts_to_save=['train.py', 'mem_transformer.py'],
                         debug=args.debug,
                         world_size=args.world_size)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
if args.fp16:
    assert args.cuda is True
    from apex import amp
    print(f'Start use apex to train !!!')
elif args.autocast:
    assert torch.__version__ >= '1.6'
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    print(f'Start use autocast to train !!!')
else:
    print(f'Start use FP32 to train !!!')
    from contextlib import nullcontext as autocast  # XD

# device = torch.device('cuda' if args.cuda else 'cpu')
device = "cuda:{}".format(torch.cuda.current_device())
print(f'curretn device: {device}')

###############################################################################
# Load data
###############################################################################


def collate_fn_one(batch):
    # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
    indexes = [len(it) - 1 for item in batch for it in item[0]]
    # padding和关系融合
    # p = list(range(21, 71)) + [10]
    # p = list(range(21, 71))
    # 仅仅训练性别
    p = []
    max_len = max(indexes) + len(p) + 1
    # finetune
    input_ids = [
        np.concatenate([p, it, [0] * (max_len - len(it) -len(p))]) for item in batch for it in item[0]
    ]
    # ppl_masks = [
    #     np.concatenate([[0] * (len(p) + len(item[1]) - 1), [1], [0] * (max_len - len(item[1]) - len(p))]) for item in batch
    # ]

    # 1单向，0双向
    attn_masks = [
        np.concatenate([[0] * (len(p) + len(it)), [1] * (max_len - len(it) - len(p))]) for item in batch for it in item[0]
    ]
    target_ids = [np.array([item[2][0]]) for i, item in enumerate(batch)]
    # 把labels转换成Tensor
    input_ids = torch.Tensor(input_ids).permute(1, 0).long()
    target_ids = torch.Tensor(target_ids).permute(1, 0).long()
    # ppl_masks = torch.Tensor(ppl_masks).permute(1, 0).bool()
    # 索引
    ppl_masks = torch.tensor(indexes).view(1, -1)

    attn_masks = torch.Tensor(attn_masks).permute(1, 0).bool()


    return {'data': input_ids, 'target': target_ids, 'attn_mask': attn_masks, 'ppl_masks': ppl_masks }


def collate_fn(batch):
    # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list
    max_len_text = max([len(item[0]) for item in batch])#wab
    # input_ids = []#wab
    # for text in batch:
    #     s=(max_len_text - len(text[0])) * [0]+text[0]
    #     input_ids.append(torch.tensor(s))#atten_mask summary传mems 
    input_ids = [torch.LongTensor(item[0]).unsqueeze(0) for item in batch]  # b * [1i]
    summary_ids = []
    attn_masks = []
    ppl_masks = []
    for item in batch:
        max_len = max([len(i) for i in item[1]])
        summary_id = []
        attn_mask = []
        ppl_mask = []
        for i in item[1]:
            s = i + (max_len - len(i)) * [0]
            summary_id.append(s)
            s = len(i) * [0] + (max_len - len(i)) * [1]
            attn_mask.append(s)
            s = len(i) - 1
            ppl_mask.append(s)
        summary_ids.append(torch.LongTensor(summary_id).unsqueeze(0))
        attn_masks.append(torch.LongTensor(attn_mask).unsqueeze(0))
        ppl_masks.append(torch.LongTensor(ppl_mask).unsqueeze(0))
    # summary_ids = torch.Tensor(summary_ids).long()
    # attn_masks = torch.Tensor(attn_masks).long()
    # ppl_masks = torch.Tensor(ppl_masks).long()  #wab
    #input_ids = torch.Tensor(input_ids).long()
    # summary_ids = [item[1] for item in batch for i in item[1]]
    target_ids = [torch.LongTensor(item[2]).unsqueeze(0) for item in batch]#chunk_input 在dim=0切
    # target_ids = torch.Tensor(target_ids).long() #wab
    return {'data': input_ids, 'target': target_ids, 'summary_ids': summary_ids, 'attn_masks': attn_masks, 'ppl_masks': ppl_masks}

corpus = get_lm_corpus(args.data,
                       args.dataset,
                       vocab_file=args.vocab_file,
                       workers=args.workers,
                       namepostfix=args.namepostfix)
ntokens = len(corpus.vocab)

eval_batch_size = args.valid_batch_size

tr_iter = list(zip(*corpus.train))

train_sampler = DistributedSampler(tr_iter)
tr_iter = DataLoader(tr_iter,
                     batch_size=args.batch_size,
                     shuffle=False,
                     pin_memory=True,
                     sampler=train_sampler,
                     collate_fn=collate_fn,
                     drop_last=True)
print(f'tr_iter len = {len(tr_iter)}')
if args.do_valid:
    # 开发集
    va_iter = list(zip(*corpus.valid))
    val_sampler = DistributedSampler(va_iter)
    va_iter = DataLoader(va_iter,
                         batch_size=args.valid_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         sampler=val_sampler,
                         collate_fn=collate_fn,
                         drop_last=True)
    print(f'va_iter len = {len(va_iter)}')

    va_iter2 = list(zip(*corpus.valid2))

    val_sampler = DistributedSampler(va_iter2)

    va_iter2 = DataLoader(va_iter2,
                         batch_size=args.valid_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         sampler=val_sampler,
                         collate_fn=collate_fn,
                         drop_last=True)
    print(f'va_iter2 len = {len(va_iter2)}')

    va_iter3 = list(zip(*corpus.valid3))
    val_sampler = DistributedSampler(va_iter3)
    va_iter3 = DataLoader(va_iter3,
                         batch_size=args.valid_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         sampler=val_sampler,
                         collate_fn=collate_fn,
                         drop_last=True)
    print(f'va_iter3 len = {len(va_iter3)}')

    tr_va = list(zip(*corpus.train_valid))
    val_sampler = DistributedSampler(tr_va)
    tr_va = DataLoader(tr_va,
                         batch_size=args.valid_batch_size,
                         shuffle=False,
                         pin_memory=True,
                         sampler=val_sampler,
                         collate_fn=collate_fn,
                         drop_last=True)
    print(f'tr_va len = {len(tr_va)}')

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b']
    if args.dataset == 'wt103':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_s_bias'):
            init_weight(m.r_s_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)
    elif classname.find('RelPartialLearnableMultiHeadAttn') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_s_bias'):
            init_weight(m.r_s_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


model = MemTransformerLM(ntokens,
                         args.n_layer,
                         args.n_head,
                         args.d_model,
                         args.d_head,
                         args.d_inner,
                         args.dropout,
                         args.dropatt,
                         tie_weight=args.tied,
                         d_embed=args.d_embed,
                         div_val=args.div_val,
                         tie_projs=tie_projs,
                         pre_lnorm=args.pre_lnorm,
                         tgt_len=args.tgt_len,
                         ext_len=args.ext_len,
                         mem_len=args.mem_len,
                         cutoffs=cutoffs,
                         same_length=args.same_length,
                         attn_type=args.attn_type,
                         clamp_len=args.clamp_len,
                         sample_softmax=args.sample_softmax,
                         untie_r=args.untie_r,
                         with_token=args.with_token,
                         add_lnorm=args.add_lnorm,
                         code_type=args.code_type)
model.apply(weights_init)
model.word_emb.apply(weights_init)

if args.restart:
    with open(os.path.join(args.restart_dir, args.checkpoint_name), 'rb') as f:
        state_dict = torch.load(f, map_location='cpu')
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict, strict=False)
    print("Recover model paras with {}".format(args.restart_dir))

# relation_model_state = '/nas/lishengping/xl-models/prompt/prompt_relation_big_20220301/checkpoint-2-40000.pt'
# relation_model_state = '/nas/lishengping/xl-models/prompt/combine_prompt_relation_filter_20220303/checkpoint-4-20000.pt'
# relation_model_state = '/nas/lishengping/xl-models/prompt/combine_prompt_relation_filter_20220308/checkpoint-4-20000.pt'

# # padding_model_state = '/nas/lishengping/xl-models/prompt/padding_prompt_50un_20220223/checkpoint-3-140000.pt'
# padding_model_state = '/nas/lishengping/xl-models/prompt/padding_prompt_50un_len800_lr5e3_20220303/checkpoint-8-76000.pt'
# padding_model_state = '/nas/lishengping/xl-models/prompt/padding_prompt_50un_len800_lr5e3_20220303/checkpoint-8-76000.pt'
# padding_model_state = '/nas/lishengping/xl-models/prompt/combine_general_sex_prompt_7000.pt'

# # 加载padding_text prompt参数
# with open(padding_model_state, 'rb') as f:
#     print(f'加载 padding text prompt 参数')
#     padding_prompt_vec = torch.load(f, map_location='cpu')
#     # model.word_emb.padding_prompt.weight.data[21:71] = padding_prompt_vec['padding_prompt_vec'][21:71]
#     model.word_emb.padding_prompt.weight.data[21:100] = padding_prompt_vec['padding_prompt_vec'][21:100]

# with open(relation_model_state, 'rb') as f:
#     print(f'加载 relation text prompt 参数')
#     relation_prompt_vec = torch.load(f, map_location='cpu')
#     model.relation_prompt.weight.data[:21] = relation_prompt_vec['relation_prompt_vec'][:21]

# padding prompt
for k, v in model.named_parameters():
    print(f'k: {k} v: {v.shape}')
    if 'padding' not in k and 'reward' not in k:
        v.requires_grad_(False)
    else:
        print(f'带有梯度的参数：{k} shape: {v.shape}')

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.multi_gpu:
    model = model.to(device)

# optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        # print(f'model.parameters(): {model.parameters()}')
        param_optimizer = list(model.named_parameters())
        param_groups = [{'params':[v for k, v in param_optimizer if 'padding' in k], 
            'weight_decay_rate': 0.01,
            'lr': 0.05
            },
            {'params':[v for k, v in param_optimizer if 'reward' in k], 
            'weight_decay_rate': 0.01,
            'lr': 0.001
           }
            ]

        optimizer = optim.Adam(param_groups)
        # optimizer = optim.Adam(model.parameters())


elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parametersfo(), lr=args.lr)

# scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_step, eta_min=args.eta_min)    # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sparse, args.max_step, eta_min=args.eta_min)    # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                else step / (args.warmup_step ** 1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_rate,
                                                     patience=args.patience,
                                                     min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                                                                factor=args.decay_rate,
                                                                patience=args.patience,
                                                                min_lr=args.lr_min)
elif args.scheduler == 'constant':

    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        return 1

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
logging(f'optimizer: {optimizer}')

if args.fp16:    # apex
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.optim_level)

if args.restart:
    print('Recover optim papras from {}'.format(args.restart_dir))
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f, map_location="cpu")
            try:
                optimizer.load_state_dict(opt_state_dict)
            except Exception as ValueError:
                print('incompatible shape, couldn\'t loaded, start from scratch.')
    else:
        print('Optimizer was not saved. Start from scratch.')


if args.world_size > 1:
    model = DistributedDataParallel(model,
                                    device_ids=[args.rank],
                                    output_device=args.rank,
                                    find_unused_parameters=True)
else:
    model = model.to(device)

logging('=' * 100)
args_string = "Namespace("
for k, v in args.__dict__.items():
    args_string += "{}={}, ".format(k, v)
args_string += ")"
logging(args_string)
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################
def show_word(data):
    global corpus
    print(corpus.vocab.convert_ids_to_tokens(data.tolist()))


def evaluate(model, eval_iter, valid_type=''):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    # Evaluation
    n_tokens, token_loss = 0, 0
    content_loss, content_tokens = 0, 0
    total_right = 0
    with torch.no_grad():
        mems = tuple()
        for i, data in enumerate(eval_iter):#测试i
            #input_chunks = chunk_inputs(data, 1, device)#wab
            data = move_to_cuda(data, device)#wab
            #input_chunks = chunk_inputs(data, 1)
            assert all(type(v) == list and len(v) == 1 for v in data.values())
            inputs = {k: v[0] for k, v in data.items()}
            # inputs = input_chunks[0]
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            if not args.with_token:
                inputs['token_ids'] = None
            context = inputs['data'].permute(1, 0)
            ppl_masks = inputs['ppl_masks'].permute(1, 0) 
            # print(f'ppl_masks: {ppl_masks.shape}')
            summary_ids = inputs['summary_ids'].view(-1, inputs['summary_ids'].size(-1)).permute(1, 0)
            target = inputs['target']
            attn_mask = torch.zeros_like(context)
            summary_mems = model(context, None, None, attn_mask=None, context=True)
            #mems = model.init_mems()#wab
            #_, summary_mems = model._forward(context)  #wab
            batch_summary_mems = [m.expand(m.size(0), summary_ids.size(1), m.size(-1)) for m in summary_mems]
            # ret = model(summary_ids,token_ids=None, target=target, mems=batch_summary_mems, attn_mask=inputs['attn_masks'], ppl_masks=ppl_masks)
            ret = model(summary_ids, token_ids=None, target=target, mems=batch_summary_mems, attn_mask=None, ppl_masks=ppl_masks, context=False)


            loss, mems, reward = ret[0], ret[1], ret[2]
            pred = torch.argmax(reward, dim=-1)

            cur_right = (target == pred).sum()
            # 非人名loss
            #cur_content_loss = loss.masked_select(ppl_masks.bool())
            #cur_content_tokens = ppl_masks.view(-1).sum()
            #cur_content_loss = cur_content_loss.sum().float().item()
            # 非关系文本loss
            #loss = loss.masked_select(ppl_masks.bool())
            #cur_n_tokens = ppl_masks.view(-1).sum()
            #cur_token_loss = loss.sum().float().item()


            cur_content_tokens = len(ppl_masks.view(-1)) / 3
            cur_content_loss = loss.sum().float().item()

            # 非关系文本loss
            cur_n_tokens = cur_content_tokens
            cur_token_loss = cur_content_loss
            if args.world_size > 1:
                cur_n_tokens = torch.Tensor([cur_n_tokens]).to(loss.device)
                cur_token_loss = torch.Tensor([cur_token_loss]).to(loss.device)
                cur_right = torch.Tensor([cur_right]).to(loss.device)

                cur_content_tokens = torch.Tensor([cur_content_tokens]).to(loss.device)
                cur_content_loss = torch.Tensor([cur_content_loss]).to(loss.device)

                reduced_param = torch.cat((cur_token_loss.view(1), cur_n_tokens.view(1),
                                           cur_content_loss.view(1), cur_content_tokens.view(1), cur_right, ))

                dist.all_reduce(reduced_param.data)
                cur_token_loss = reduced_param[0].item()
                cur_n_tokens = reduced_param[1].item()
                cur_content_loss = reduced_param[2].item()
                cur_content_tokens = reduced_param[3].item()
                cur_right = reduced_param[4].item()


            token_loss += cur_token_loss
            n_tokens += cur_n_tokens
            total_right += cur_right

            content_loss += cur_content_loss
            content_tokens += cur_content_tokens

    # logging(f'{valid_type} eval acc： {total_right / (i + 1) /  args.world_size / args.valid_batch_size}')
    acc = total_right / (i + 1) /  args.world_size / args.valid_batch_size

    model.train()
    return token_loss / n_tokens, content_loss / content_tokens, acc


def evaluate_one(model, eval_iter, valid_type=''):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    n_tokens, token_loss = 0, 0
    content_tokens, content_loss = 0, 0
    total_right = 0
    with torch.no_grad():
        mems = tuple()
        for i, pack_sample in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            inputs = pack_sample
            inputs = move_to_cuda(inputs, device)
            ppl_masks = inputs["ppl_masks"]
            label = inputs['target']
            if not args.with_token:
                inputs['token_ids'] = None
            # 名字处为0
            ret = model(**inputs, mems=mems)
            loss, mems, reward = ret[0], ret[1], ret[2]
            pred = torch.argmax(reward, dim=-1)
            cur_right = (label == pred).sum()
            # 非人名loss
            # cur_content_loss = loss.masked_select(ppl_masks.bool())
            cur_content_tokens = len(ppl_masks.view(-1)) / 3
            cur_content_loss = loss.sum().float().item()

            # 非关系文本loss
            cur_n_tokens = cur_content_tokens
            cur_token_loss = cur_content_loss

            if args.world_size > 1:
                cur_n_tokens = torch.Tensor([cur_n_tokens]).to(loss.device)
                cur_token_loss = torch.Tensor([cur_token_loss]).to(loss.device)
                cur_right = torch.Tensor([cur_right]).to(loss.device)

                cur_content_tokens = torch.Tensor([cur_content_tokens]).to(loss.device)
                cur_content_loss = torch.Tensor([cur_content_loss]).to(loss.device)

                reduced_param = torch.cat((cur_token_loss.view(1), cur_n_tokens.view(1),
                                           cur_content_loss.view(1), cur_content_tokens.view(1), cur_right, ))

                dist.all_reduce(reduced_param.data)
                cur_token_loss = reduced_param[0].item()
                cur_n_tokens = reduced_param[1].item()
                cur_content_loss = reduced_param[2].item()
                cur_content_tokens = reduced_param[3].item()
                cur_right = reduced_param[4].item()


            token_loss += cur_token_loss
            n_tokens += cur_n_tokens
            total_right += cur_right
            content_loss += cur_content_loss
            content_tokens += cur_content_tokens

    logging(f'{valid_type} eval acc： {total_right / (i + 1) /  args.world_size / args.valid_batch_size}')
    model.train()
    return token_loss / n_tokens, content_loss / content_tokens


def train_one_step_one(model, optimizer, inputs, mems, args):
    ppl_masks = inputs['ppl_masks']
    if not args.with_token:
        inputs['token_ids'] = None
    if args.autocast:
        with autocast():
            ret = model(**inputs, mems=mems)
    else:
        ret = model(**inputs, mems=mems)
    loss, mems, reward = ret[0], ret[1], ret[2]
    label = inputs['target'].view(-1)
    pred = torch.argmax(reward, dim=-1)
    right = (label == pred).sum()
    # loss = loss.masked_select(ppl_masks.bool())
    # print(loss.view(-1).tolist())
    cur_n_tokens = len(ppl_masks.view(-1))
    cur_token_loss = loss.sum().float()
    if torch.isnan(cur_token_loss):
        print(f'出现error loss: {loss}')
        cur_token_loss = torch.tensor([2], device=cur_token_loss.device) * cur_n_tokens
    cur_token_loss = cur_token_loss.item()
    loss = loss.float().type_as(loss).mean() / args.batch_chunk
    if args.autocast:
        scaler.scale(loss).backward()
    elif args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return cur_token_loss, cur_n_tokens, loss, mems, [right, len(label)]


def train_one_step(model, optimizer, inputs, mems, args):
    if not args.with_token:
        inputs['token_ids'] = None
    
    with autocast():
        context = inputs['data'].permute(1, 0)
        ppl_masks = inputs['ppl_masks'].permute(1, 0)  # b3->3b
        summary_ids = inputs['summary_ids'].view(-1, inputs['summary_ids'].size(-1)).permute(1, 0)
        target = inputs['target']
        #print(f'inputs: {inputs}')
        # print(f'summary_ids: {summary_ids.shape}')
        context1 = context
        summary_mems = model(context1, None, None, context=True)
        batch_summary_mems = [m.expand(m.size(0), summary_ids.size(1), m.size(-1)) for m in summary_mems]
        ret = model(summary_ids, token_ids=None, target=target, mems=batch_summary_mems, ppl_masks=ppl_masks,context=False)
        
    loss, mems, reward = ret[0], ret[1], ret[2]
    
    pred = torch.argmax(reward, dim=-1)
    cur_right = (target == pred).sum().item()
    # 非人名loss
    cur_content_loss = loss.masked_select(ppl_masks.bool())

    cur_content_tokens = ppl_masks.view(-1).sum()
    cur_content_loss = cur_content_loss.sum().float().item()

    # 非关系文本loss
    loss = loss.masked_select(ppl_masks.bool())
    cur_token_loss = loss.sum().float().item()
    cur_n_tokens = ppl_masks.view(-1).sum()
    loss = loss.float().type_as(loss).mean() / args.batch_chunk
    if args.autocast:
        scaler.scale(loss).backward()
    elif args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return cur_token_loss, cur_n_tokens, loss, mems, [cur_right, len(target)]


def train(model, device):
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time, args
    token_loss = 0
    n_tokens = 0
    skip_step = args.skip_step
    resume_train = args.restart
    args.restart = False
    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    nbatch = len(train_iter)
    train_batch_size = args.batch_size * args.world_size
    right = 0
    total = 0
    for batch, inputs in enumerate(train_iter):
        if resume_train and train_step % nbatch < skip_step:
            train_step += 1
            continue
        else:
            resume_train = False
        inputs = move_to_cuda(inputs, device)

        model.zero_grad()
        if args.batch_chunk > 1:
            # data_chunks = torch.chunk(data, args.batch_chunk, 1)
            # if token_ids:
            #     token_ids_chunks = torch.chunk(token_ids, args.batch_chunk, 1)
            # target_chunks = torch.chunk(target, args.batch_chunk, 1)
            #input_chunks = chunk_inputs(inputs, args.batch_chunk)
            input_chunks = inputs
            cur_token_loss = 0
            cur_n_tokens = 0
            final_chunk = min(args.batch_chunk, len(input_chunks))
            for i in range(final_chunk):
                # inputs_i = input_chunks[i]
                inputs_i = {k: v[i] for k, v in input_chunks.items()}
                if args.world_size > 1:
                    with model.no_sync():
                        tmp_token_loss, tmp_n_tokens, loss, mems[i], mes = train_one_step(
                            model, optimizer, inputs_i, mems[i], args)
                        cur_token_loss += tmp_token_loss
                        cur_n_tokens += tmp_n_tokens
                        right += mes[0]
                        total += mes[1]

                else:
                    tmp_token_loss, tmp_n_tokens, loss, mems[i], mes = train_one_step(
                        model, optimizer, inputs_i, mems[i], args)
                    cur_token_loss += tmp_token_loss
                    cur_n_tokens += tmp_n_tokens

        else:
            cur_token_loss, cur_n_tokens, loss, mems, mes = train_one_step(model, optimizer, inputs,
                                                                      mems, args)
        # print(
        #     f'the {batch} batch || loss: {loss * args.batch_chunk} train acc: {right / total}'
        # )
        if args.world_size > 1:
            cur_n_tokens = torch.Tensor([cur_n_tokens]).to(loss.device)
            cur_token_loss = torch.Tensor([cur_token_loss]).to(loss.device)
            reduced_param = torch.cat((cur_token_loss.view(1), cur_n_tokens.view(1)))
            dist.all_reduce(reduced_param.data)
            cur_token_loss = reduced_param[0].item()
            cur_n_tokens = reduced_param[1].item()
        token_loss += cur_token_loss
        n_tokens += cur_n_tokens
        torch.cuda.empty_cache()
        if args.autocast:
            scaler.unscale_(optimizer)    # 当多个优化器的时候，每个step仅需要调用一次，不然会报错。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)

        # lsp
        if args.autocast:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if args.sample_softmax > 0:
            optimizer_sparse.step()
        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler in ['inv_sqrt', 'constant']:
            scheduler.step(train_step)

        if train_step % args.log_interval == 0:
            # cur_loss = train_loss / args.log_interval
            per_token_loss = token_loss / n_tokens * 3
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d}, total-step {} |cur-batches {} / {} | batch-size {} | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, nbatch, train_batch_size, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, per_token_loss)
            ppl_str = '{:9.3f}'.format(math.exp(per_token_loss))
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(per_token_loss / math.log(2))
            else:
                log_str += ' | ppl {}'.format(ppl_str)
            logging(log_str)
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            if args.do_valid:
                # valid
                val_loss, content_loss, acc = evaluate(model, va_iter, 'mghl')
                val_loss2, content_loss, acc2 = evaluate(model, va_iter2, 'mgml')
                val_loss3, content_loss, acc3 = evaluate(model, va_iter3, 'mix')
                train_loss, content_loss, train_acc = evaluate(model, tr_va, 'train')
                logging(f'train acc {train_acc} | mghl acc {acc} | mgml acc {acc2} | mix acc {acc3} | mean acc {(acc + acc2 + acc3) / 3}')

                logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                        '| mghl loss {:5.4f} | mgml loss {:5.4f} | mix loss {:5.4f} | train loss {:5.4f}'.format(train_step // args.eval_interval, train_step, (time.time() - eval_start_time), val_loss, val_loss2, val_loss3, train_loss)
                if args.dataset in ['enwik8', 'text8']:
                    log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                else:
                    log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                logging(log_str)
                logging('-' * 100)

            if dist.get_rank() == 0:
                # if not best_val_loss or val_loss < best_val_loss:
                save_model_name = "checkpoint-{}-{}.pt".format(epoch, train_step)
                print(f'===== model save to {args.work_dir} ======')
                save_model_name = os.path.join(args.work_dir, save_model_name)
                print(f'开始保存 ’{save_model_name}‘ 模型')
                state_save = {}
                for k, v in model.state_dict().items():
                    if 'reward' in k or 'padding' in k:
                        state_save[k] = v
                # if args.fp16:
                #     print(f'fp16 模型保存...')
                #     state_save = {
                #         "model": model.state_dict(),
                #     }
                # else:
                #     state_save = {"model": model.state_dict()}
                    
                if torch.__version__ >= '1.6':
                    torch.save(state_save, save_model_name, _use_new_zipfile_serialization=False)
                else:
                    torch.save(state_save, save_model_name)

                best_val_loss = val_loss
        if train_step >= args.max_step:
            break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None
log_start_time = time.time()
eval_start_time = time.time()
# At any point you can hit Ctrl + C to break out of training early.
skip_epoch = args.skip_epoch
start_epoch = skip_epoch + 1

train_step = skip_epoch * len(tr_iter)

try:
    logging(str(datetime.now()))
    # val_loss, content_loss, acc = evaluate(model, va_iter, 'mghl')
    # val_loss2, content_loss, acc2 = evaluate(model, va_iter2, 'mgml')
    # val_loss3, content_loss, acc3 = evaluate(model, va_iter3, 'mix')
    # train_loss, content_loss, train_acc = evaluate(model, tr_va, 'train')
    # logging(f'train acc {train_acc} | mghl acc {acc} | mgml acc {acc2} | mix acc {acc3} | mean acc {(acc + acc2 + acc3) / 3}')

    # logging(f'origin loss: {val_loss} ppl: {math.exp(val_loss)}')
    # logging(f'origin loss2: {val_loss2} ppl: {math.exp(val_loss2)}')
    # logging(f'origin loss3: {val_loss3} ppl: {math.exp(val_loss3)}')
    # logging(f'origin train loss: {val_loss3} ppl: {math.exp(train_loss)}')

    for epoch in itertools.count(start=start_epoch):
        train(model, device)
        if train_step == args.max_step:
            logging(str(datetime.now()))
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')
