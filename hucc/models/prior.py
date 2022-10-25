# Code in this file is based on modeling code from
# https://github.com/openai/jukebox, originally released under a Noncommercial
# Use License (see https://github.com/openai/jukebox/blob/master/LICENSE).
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
#

import functools
import gc
import logging
import math
import time

import numpy as np
import torch as th
import torch.distributed as dist
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

# Import FusedLayerNorm if we have apex, otherwise use regular LayerNorm
try:
    from apex.normalization import FusedLayerNorm

    print("Using apex FusedLayerNorm")
except ImportError:
    from torch.nn import LayerNorm as FusedLayerNorm

from hucc.utils import dim_select

log = logging.getLogger(__name__)


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


def empty_cache():
    gc.collect()
    th.cuda.empty_cache()


def calculate_strides(strides, downs):
    return [stride**down for stride, down in zip(strides, downs)]


def get_range(x):
    # Can return tqdm for rank 0
    return x


# Simple gradient checkpointing. Works with distributed data parallel
def checkpoint(func, inputs, params, flag):
    if flag:
        args = inputs + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with th.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del output_tensors
        return (None, None) + input_grads


class ResConv1DBlock(nn.Module):
    def __init__(
        self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0
    ):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            # if dist.get_rank() == 0:
            print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x,), block.parameters(), True)
            return x
        else:
            return self.model(x)


class DecoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_decoder_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation,
                        checkpoint_res=checkpoint_res,
                    ),
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if i == (down_t - 1) else width,
                        filter_t,
                        stride_t,
                        pad_t,
                    ),
                )
                blocks.append(block)
        else:
            assert stride_t == 1
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            block = nn.Sequential(
                Resnet1D(
                    width,
                    depth,
                    m_conv,
                    dilation_growth_rate,
                    dilation_cycle,
                    zero_out=zero_out,
                    res_scale=res_scale,
                    reverse_dilation=reverse_decoder_dilation,
                    checkpoint_res=checkpoint_res,
                ),
                nn.ConvTranspose1d(
                    width, input_emb_width, kernel_size=3, stride=1, padding=1
                ),
            )
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Conditioner(nn.Module):
    def __init__(
        self,
        input_shape,
        continuous,
        bins,
        down_t,
        stride_t,
        out_width,
        init_scale,
        zero_out,
        res_scale,
        **block_kwargs,
    ):
        super().__init__()
        self.x_shape = input_shape
        self.continuous = continuous

        # Embedding
        self.width = out_width
        if not self.continuous:
            self.x_emb = nn.Embedding(bins, out_width)
        else:
            self.x_emb = nn.Linear(bins, out_width, bias=False)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)

        # Conditioner
        self.cond = DecoderConvBlock(
            self.width,
            self.width,
            down_t,
            stride_t,
            **block_kwargs,
            zero_out=zero_out,
            res_scale=res_scale,
        )
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0, 2, 1)  # NTC -> NCT
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)  # NCT -> NTC
        return x

    def forward(self, x, x_cond=None):
        if isinstance(x, D.Normal):
            x = x.mean

        N = x.shape[0]
        if not self.continuous:
            assert_shape(x, (N, *self.x_shape))
        else:
            assert x.shape[2] == self.x_shape[0]
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        # Embed x
        if not self.continuous:
            x = x.long()
            x = self.x_emb(x)
            assert_shape(x, (N, *self.x_shape, self.width))
        else:
            x = self.preprocess(x)
            x = self.x_emb(x)
        x = x + x_cond

        # Run conditioner
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x


class SimpleConditioner(nn.Module):
    def __init__(
        self,
        input_shape,
        continuous,
        bins,
        down_t,
        stride_t,
        out_width,
        init_scale,
        zero_out,
        res_scale,
        past_ctx,
        **block_kwargs,
    ):
        super().__init__()
        self.x_shape = input_shape
        self.continuous = continuous

        # Embedding
        self.width = out_width
        if not self.continuous:
            self.x_emb = nn.Embedding(bins, out_width)
        else:
            self.x_emb = nn.Linear(bins, out_width, bias=False)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)

        self.repeat = 2**down_t
        hid = block_kwargs['width']
        if past_ctx > 0:
            # Shifted conv [-past_ctx back, 1 in future]
            ks = past_ctx + 2
            padding = past_ctx
        else:
            ks = 1
            padding = 0
        self.cond1 = nn.Sequential(
            nn.Conv1d(self.width, hid, ks, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.cond2 = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_width),
        )
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0, 2, 1)  # NTC -> NCT
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)  # NCT -> NTC
        return x

    def forward(self, x, x_cond=None):
        if isinstance(x, D.Normal):
            x = x.mean

        N = x.shape[0]
        if not self.continuous:
            assert_shape(x, (N, *self.x_shape))
        else:
            # assert x.shape[2] == self.x_shape[0]
            pass  # not important?
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        # Embed x
        if not self.continuous:
            x = x.long()
            x = self.x_emb(x)
            assert_shape(x, (N, *self.x_shape, self.width))
        else:
            x = self.preprocess(x)
            x = self.x_emb(x)

        x = x + x_cond

        # Run conditioner
        x = self.preprocess(x)
        x = self.cond1(x)
        # Shifted conv
        if self.cond1[0].padding[0] > 1:
            x = x[:, :, : -self.cond1[0].padding[0] + 1]
        x = self.postprocess(x)
        x = self.cond2(x)
        x = self.ln(x)

        # Upsample by repetition
        x = x.repeat_interleave(self.repeat, 1)
        return x


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with th.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del output_tensors
        return (None, None) + input_grads


class EmptyLabeller:
    def get_label(
        self,
        artist=None,
        genre=None,
        lyrics=None,
        total_length=None,
        offset=None,
    ):
        y = np.array([], dtype=np.int64)
        info = dict(artist="n/a", genre="n/a", lyrics=[], full_tokens=[])
        return dict(y=y, info=info)

    def get_batch_labels(self, metas, device='cpu'):
        ys, infos = [], []
        for meta in metas:
            label = self.get_label()
            y, info = label['y'], label['info']
            ys.append(y)
            infos.append(info)

        ys = t.stack([t.from_numpy(y) for y in ys], dim=0).to(device).long()
        assert ys.shape[0] == len(metas)
        assert len(infos) == len(metas)
        return dict(y=ys, info=infos)


def get_normal(*shape, std=0.01):
    w = th.empty(shape)
    nn.init.normal_(w, std=std)
    return w


def roll(x, n):
    return th.cat((x[:, -n:], x[:, :-n]), dim=1)


def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [
        *[chunk_size] * (n_passes - 1),
        (length - 1) % chunk_size + 1,
    ]
    assert sum(chunk_sizes) == length
    return chunk_sizes


class LayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(
                input.float(),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).type_as(input)
        else:
            return super(LayerNorm, self).forward(input.float()).type_as(input)


def gelu(x):
    return (
        0.5
        * x
        * (1 + th.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * th.pow(x, 3))))
    )


def swish(x):
    return x * th.sigmoid(x)


@th.jit.script
def quick_gelu(x):
    return x * th.sigmoid(1.702 * x)


@th.jit.script
def quick_gelu_bwd(x, grad_output):
    sig = th.sigmoid(1.702 * x)
    return grad_output * sig * (1.702 * x * (1 - sig) + 1.0)


class QuickGelu(th.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return quick_gelu(x)

    @staticmethod
    def backward(ctx, grad_output):
        return quick_gelu_bwd(ctx.saved_tensors[0], grad_output)


def memory_efficient_quick_gelu(x):
    return QuickGelu.apply(x)


ACT_FNS = {
    'relu': F.relu,
    'swish': swish,
    'gelu': gelu,
    'quick_gelu': memory_efficient_quick_gelu,  # quick_gelu
}


def _move_to_gpu_and_convert_conv_weights_to_fp16(l):
    l.cuda()
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.half()


def _convert_conv_weights_to_fp32(l):
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.float()


def _convert_conv_weights_to_fp16(l):
    if isinstance(l, Conv1D):
        l.w.data = l.w.data.half()


def _convert_embedding_weights_to_fp16(l):
    if isinstance(l, th.nn.Embedding):
        l.weight.data = l.weight.data.half()


def _convert_embedding_weights_to_fp32(l):
    if isinstance(l, t.nn.Embedding):
        l.weight.data = l.weight.data.float()


class Conv1D(nn.Module):
    def __init__(self, n_in, n_out, zero_out=False, init_scale=1.0):
        super(Conv1D, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if zero_out:
            w = th.zeros(n_in, n_out)
        else:
            w = th.empty(n_in, n_out)
            nn.init.normal_(w, std=0.02 * init_scale)
        b = th.zeros(n_out)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def forward(self, x):
        size_out = (*x.size()[:-1], self.n_out)
        x = th.addmm(
            self.b.type_as(x), x.view(-1, x.size(-1)), self.w.type_as(x)
        )  # If x if float then float else half
        x = x.view(*size_out)
        return x


# For large contexts, mask's can take up memory, so you can make a single saved mask for all layers
class Mask(nn.Module):
    def __init__(self, n_ctx):
        super().__init__()
        self.register_buffer(
            'b', th.tril(th.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        )

    def forward(self, w):
        w = w * self.b + -1e9 * (
            1 - self.b
        )  # For fp16 do w = w.float().masked_fill(self.b, float('-inf')
        return w


def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    # assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check
    assert (top_k == 0) or (top_p == 0.0)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < th.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = th.sort(logits, descending=True, dim=-1)
        cumulative_probs = th.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove = th.zeros_like(logits, dtype=th.uint8).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def repeat(x, n, dim):
    if dim == -1:
        dim = len(x.shape) - 1
    return (
        x.view(
            int(np.prod(x.shape[: dim + 1])),
            1,
            int(np.prod(x.shape[dim + 1 :])),
        )
        .repeat(1, n, 1)
        .view(*x.shape[:dim], n * x.shape[dim], *x.shape[dim + 1 :])
    )


def get_mask(mask, q_l, kv_l, blocks, spread, device, sample, sample_t):
    # returns a mask of shape 1 x 1 x q_l x kv_l or None if masking is not needed.
    if mask is None or q_l == 1:
        return None
    offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
    if mask == 'autoregressive':
        # Masked dense
        mask = th.ones(q_l, kv_l, device=device).tril(offset)
    elif mask == 'summary':
        # Maskedh summary
        mask = (
            F.pad(
                th.ones(q_l, q_l, device=device)
                .tril()
                .view(q_l, blocks, q_l // blocks)[:, :-1, -kv_l // blocks :],
                (0, 0, 1, 0),
                value=1,
            )
            .contiguous()
            .view(q_l, kv_l)
        )
    elif mask == 'prime':
        mask = th.ones(q_l, kv_l, device=device).tril(offset)
    return mask.view(1, 1, q_l, kv_l)


class FactoredAttention(nn.Module):
    def __init__(
        self,
        n_in,
        n_ctx,
        n_state,
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        checkpoint_attn=0,
        attn_func=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx  # NOTE: n_ctx could be different within operations. This is complete n_ctx
        self.n_state = n_state
        assert n_state % n_head == 0
        self.n_head = n_head
        self.scale = scale
        self.mask = mask
        if attn_func == 6:
            self.c_attn = Conv1D(n_in, n_state, init_scale=init_scale)
            self.c_enc_kv = Conv1D(n_in, n_state * 2, init_scale=init_scale)
        else:
            self.c_attn = Conv1D(n_in, n_state * 3, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.attn_dropout = (
            nn.Dropout(attn_dropout) if attn_dropout > 0.0 else lambda x: x
        )
        self.resid_dropout = (
            nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x
        )

        # Sequence of length l is factored as [blocks, l // blocks]
        self.attn_func = attn_func
        self.qkv, self.attn, self.attn_mask = {
            0: (
                self.factored_qkv,
                self.dense_attn,
                'autoregressive',
            ),  # Attend to all positions
            1: (
                self.factored_qkv,
                self.block_attn,
                'autoregressive',
            ),  # Attend to your block
            2: (
                self.factored_qkv,
                self.transpose_block_attn,
                'autoregressive',
            ),  # Attend to transpose block
            3: (
                self.factored_qkv,
                self.prev_block_attn,
                None,
            ),  # Attend to previous block
            4: (
                self.factored_qkv,
                self.summary_attn,
                'summary',
            ),  # Attend to last position of each block
            5: (self.factored_qkv, self.summary_spread_attn, 'summary'),
            6: (self.decode_qkv, self.decode_attn, None),
            7: (self.prime_qkv, self.prime_attn, 'prime'),
        }[
            attn_func
        ]  # Attend to last k position of each block

        self.blocks = blocks
        self.spread = spread
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.checkpoint_attn = (
            checkpoint_attn  # 0: None, 1: Attn after heads split, 2: Attn
        )

        self.sample_t = 0
        self.cache = {}
        self.encoder_dims = encoder_dims
        self.prime_len = prime_len
        self.record_attn = False
        self.w = None

    def _attn(self, q, k, v, sample):
        scale = 1.0 / math.sqrt(math.sqrt(self.n_state // self.n_head))
        if self.training:
            w = th.matmul(q * scale, k * scale)
        else:
            w = th.matmul(q, k)
            w.mul_(scale * scale)
        wtype = w.dtype
        w = w.float()

        if self.mask:
            # Generate appropriate mask to mask out all positions before current
            # Might take up lot of memory for dense, so can cache it
            mask = get_mask(
                self.attn_mask,
                q.size(-2),
                k.size(-1),
                self.blocks,
                self.spread,
                w.device,
                sample,
                self.sample_t,
            )
            if mask is not None:
                # print(mask)
                w = w * mask + -1e9 * (1 - mask)
            w = F.softmax(w, dim=-1).type(wtype)
        else:
            w = F.softmax(w, dim=-1).type(wtype)
        if self.record_attn:
            self.w = w  # .float().cpu().numpy()
            if self.attn_func == 7:
                # only keep music queries and lyrics keys/values
                self.w = self.w[:, :, self.prime_len :, : self.prime_len]
        w = self.attn_dropout(w)
        a = th.matmul(w, v)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = (*x.size()[:-2], x.size(-2) * x.size(-1))
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = (*x.size()[:-1], self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if self.checkpoint_attn == 1 and not sample:
            a = checkpoint(
                lambda q, k, v, s=sample: self._attn(q, k, v, s),
                (query, key, value),
                (),
                True,
            )
        else:
            a = self._attn(query, key, value, sample)
        a = self.merge_heads(a)
        return a

    def block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert (
                l == self._suff_cache_len()
            ), f"{l} != {self._suff_cache_len()}"
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            if ql < l:
                l = ql
                k = k[:, -l:].contiguous()
                v = v[:, -l:].contiguous()
            k = k.view(bs * l // block_ctx, block_ctx, d)
            v = v.view(bs * l // block_ctx, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def transpose_block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            block_l = (l - 1) % block_ctx
            k = k[:, block_l::block_ctx, :]
            v = v[:, block_l::block_ctx, :]
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = (
                q.view(bs, ql // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, ql // block_ctx, d)
            )
            k = (
                k.view(bs, l // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, l // block_ctx, d)
            )
            v = (
                v.view(bs, l // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, l // block_ctx, d)
            )
            return (
                self.dense_attn(q, k, v, sample)
                .view(bs, block_ctx, ql // block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs, ql, d)
            )

    def prev_block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert (
                l == self._suff_cache_len()
            ), f"{l} != {self._suff_cache_len()}"
            block = (l - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                assert prev_l == 0
                k = k[:, prev_l : prev_l + block_ctx, :]
                v = v[:, prev_l : prev_l + block_ctx, :]
            else:
                k = th.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
                v = th.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            k = F.pad(
                k.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :],
                (0, 0, 0, 0, 1, 0),
            ).view(bs * l // block_ctx, block_ctx, d)
            v = F.pad(
                v.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :],
                (0, 0, 0, 0, 1, 0),
            ).view(bs * l // block_ctx, block_ctx, d)
            if ql < l:
                qb = ql // block_ctx
                kb = l // block_ctx
                l = ql
                k = (
                    k.view(bs, kb, block_ctx, d)[:, -qb:]
                    .contiguous()
                    .view(bs * qb, block_ctx, d)
                )
                v = (
                    v.view(bs, kb, block_ctx, d)[:, -qb:]
                    .contiguous()
                    .view(bs * qb, block_ctx, d)
                )
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            k = F.pad(
                k[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :],
                (0, 0, 1, 0),
            )
            v = F.pad(
                v[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :],
                (0, 0, 1, 0),
            )
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = F.pad(
                k.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0)
            )  # bs, blocks, d
            v = F.pad(
                v.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0)
            )  # bs, blocks, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_spread_attn(self, q, k, v, sample):
        blocks, block_ctx, spread = (
            self.blocks,
            self.block_ctx,
            self.spread,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert False, "Not yet implemented"
            # k = F.pad(k,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # v = F.pad(v,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = (
                F.pad(
                    k.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :],
                    (0, 0, 0, 0, 1, 0),
                )
                .contiguous()
                .view(bs, blocks * spread, d)
            )  # bs, blocks * spread, d
            v = (
                F.pad(
                    v.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :],
                    (0, 0, 0, 0, 1, 0),
                )
                .contiguous()
                .view(bs, blocks * spread, d)
            )  # bs, blocks * spread, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def prime_attn(self, q, k, v, sample):
        prime_len = self._prime_len
        k = k[:, :prime_len]
        v = v[:, :prime_len]
        return self.dense_attn(q, k, v, sample)

    def decode_attn(self, q, k, v, sample):
        assert (
            k.shape[1] == v.shape[1] == self.encoder_dims
        ), f'k: {k.shape}, v: {v.shape}, enc_dims: {self.encoder_dims}'
        return self.dense_attn(q, k, v, sample)

    def factored_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 0:
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                    assert key.shape[1] % self.block_ctx == 0
                    assert query.shape[1] % self.block_ctx == 0
                assert key.shape[1] == value.shape[1]
                assert query.shape[1] <= key.shape[1]
                sample = False
            else:
                key = self.cache['key']
                value = self.cache['value']
        return query, key, value, sample

    def prime_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._prime_len:
                self._append_cache(key, value)
            if self._cache_len() > self._prime_len:
                self._slice_cache(0, self._prime_len)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
            assert (
                key.shape[1] == value.shape[1] == self._suff_cache_len()
            ), f'k: {key.shape}, v: {value.shape}, prime_dims: {self._suff_cache_len()}'
        else:
            assert (
                key.shape[1] == value.shape[1] == self.n_ctx
            ), f'k: {key.shape}, v: {value.shape}, prime_dims: {self.n_ctx}'
        assert (
            key.shape[0] == value.shape[0] == query.shape[0]
        ), f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert (
            key.shape[2] == value.shape[2] == query.shape[2]
        ), f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def decode_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is not None
        query = x
        if sample:
            if self.sample_t == 0:
                self.cache['key'], self.cache['value'] = self.c_enc_kv(
                    encoder_kv.type_as(x)
                ).chunk(2, dim=2)
            key, value = self.cache['key'], self.cache['value']
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
        assert (
            key.shape[0] == value.shape[0] == query.shape[0]
        ), f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        assert (
            key.shape[1] == value.shape[1] == self.encoder_dims
        ), f'k: {key.shape}, v: {value.shape}, enc_dims: {self.encoder_dims}'
        assert (
            key.shape[2] == value.shape[2] == query.shape[2]
        ), f'k: {key.shape}, v: {value.shape}, q: {query.shape}'
        return query, key, value, sample

    def forward(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        x = self.c_attn(x)
        query, key, value, sample = self.qkv(
            x, encoder_kv=encoder_kv, sample=sample
        )
        if self.checkpoint_attn == 2 and not sample:
            a = checkpoint(
                lambda q, k, v, s=sample: self.attn(q, k, v, s),
                (query, key, value),
                (),
                True,
            )
        else:
            a = self.attn(query, key, value, sample)
        if a.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            a = a[:, offset : offset + curr_ctx, :].contiguous()
        a = self.c_proj(a)
        return self.resid_dropout(a)

    @property
    def _prime_len(self):
        prime_len = self.prime_len
        assert prime_len is not None
        prime_blocks = (prime_len // self.blocks) + 1
        return prime_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 0:
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, x, query=False):
        l = x.shape[1]
        offset = self._offset(l) if query else 0
        n_blocks = (l + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - l - offset
        if pad == 0 and offset == 0:
            return x
        else:
            return F.pad(x, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if 'key' not in self.cache else self.cache['key'].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and
            self.sample_t reflects the 1-indexed sample location in the
            context.
        """
        if self.attn_func == 0:
            return self.sample_t
        elif self.attn_func == 1:
            return (self.sample_t - 1) % self.block_ctx + 1
        elif self.attn_func == 2:
            return self.sample_t
        elif self.attn_func == 3:
            if self.sample_t <= self.block_ctx:
                return self.sample_t
            else:
                curr_block = (self.sample_t - 1) % self.block_ctx + 1
                prev_block = self.block_ctx
                return curr_block + prev_block
        elif self.attn_func == 6:
            return self.encoder_dims
        elif self.attn_func == 7:
            return min(self.sample_t, self._prime_len)
        else:
            raise NotImplementedError()

    def _slice_cache(self, start, end=None):
        self.cache['key'] = self.cache['key'][:, start:end]
        self.cache['value'] = self.cache['value'][:, start:end]

    def _append_cache(self, key, value):
        if 'key' not in self.cache:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            old_key, old_value = key, value
            key = th.cat([self.cache['key'], key], dim=1)
            value = th.cat([self.cache['value'], value], dim=1)
            del self.cache['key']
            del self.cache['value']
            del old_key
            del old_value
            self.cache['key'] = key
            self.cache['value'] = value
        return self.cache['key'], self.cache['value']

    def del_cache(self):
        self.sample_t = 0
        if 'key' in self.cache:
            del self.cache['key']
        if 'value' in self.cache:
            del self.cache['value']
        self.cache = {}

    def check(self):
        blocks = self.blocks or 1
        spread = self.spread or 1
        bs, l, d = (4, self.n_ctx, self.n_in)
        x = th.randn(bs, l, d).cuda()
        x.requires_grad = True
        x_out = self.forward(x)  # bs, l, d
        loss = x_out.mean(dim=-1)  # bs, l
        pos = 60
        grad = th.autograd.grad(loss[2, pos], x)[0]

        assert grad.shape == (bs, l, d)
        assert (grad[:2] == 0).all()
        assert (grad[3:] == 0).all()
        assert (grad[2, (pos + 1) :] == 0).all()
        pos_grad = (th.sum(grad[2] ** 2, dim=-1) > 0).nonzero().view(-1).cpu()

        block_pos = pos - (pos % (l // blocks))
        exp_pos_grad = {
            0: th.arange(pos),
            1: th.arange(block_pos, pos),
            2: th.arange(pos % (l // blocks), pos, l // blocks),
            3: th.arange(block_pos - l // blocks, block_pos),
            4: th.arange(l // blocks - 1, pos, l // blocks),
            5: (
                (th.arange(pos) % (l // blocks) >= (l // blocks - spread))
                & (th.arange(pos) < block_pos)
            )
            .nonzero()
            .view(-1),
        }[self.attn_func]
        exp_pos_grad = th.cat([exp_pos_grad, th.tensor([pos])], dim=-1)

        assert (len(pos_grad) == len(exp_pos_grad)) and (
            pos_grad == exp_pos_grad
        ).all(), f"Expected pos grad {exp_pos_grad} got {pos_grad} for attn_func {self.attn_func} pos {pos} l {l} blocks {blocks}"

    def check_cache(self, n_samples, sample_t, fp16):
        assert self.sample_t == sample_t, f"{self.sample_t} != {sample_t}"
        if sample_t == 0:
            assert self.cache == {}
        else:
            dtype = {True: th.float16, False: th.float32}[fp16]
            l_cache = self._suff_cache_len()
            assert self.cache['key'].shape == (n_samples, l_cache, self.n_state)
            assert self.cache['value'].shape == (
                n_samples,
                l_cache,
                self.n_state,
            )
            assert (
                self.cache['key'].dtype == dtype
            ), f"Expected {dtype}, got {self.cache['key'].dtype}"
            assert (
                self.cache['value'].dtype == dtype
            ), f"Expected {dtype}, got {self.cache['value'].dtype}"

    def check_sample(self):
        th.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        prime = 5
        x = th.randn(bs, l, d).cuda()
        xs = th.chunk(x, l, dim=1)
        assert self.sample_t == 0
        assert self.cache == {}

        with th.no_grad():
            enc_l = self.encoder_dims
            encoder_kv = None
            if self.attn_func == 6:
                encoder_kv = th.randn(bs, enc_l, d).cuda()

            # Normal path
            x_out_normal = self.forward(x, encoder_kv=encoder_kv)

            # Sampling path
            x_out_sample = th.cat(
                [
                    self.forward(xs[i], encoder_kv=encoder_kv, sample=True)
                    for i in range(l)
                ],
                dim=1,
            )
        max_err = th.max(th.abs(x_out_sample - x_out_normal))
        assert (
            max_err < 1e-8
        ), f"Max sampling err is {max_err} {[i for i in range(l) if th.max(th.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

        with th.no_grad():
            x_out_normal = x_out_normal[:, :prime, :]
            # Prime sampling path
            self.del_cache()
            x_out_sample = self.forward(
                x[:, :prime, :].contiguous(), encoder_kv=encoder_kv, sample=True
            )
            self.check_cache(bs, prime, False)

        max_err = th.max(th.abs(x_out_sample - x_out_normal))
        assert (
            max_err < 1e-8
        ), f"Max prime sampling err is {max_err} {[i for i in range(prime) if th.max(th.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

    def check_chunks(self, chunk_size):
        th.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        enc_l = self.encoder_dims
        assert l % chunk_size == 0
        n_chunks = l // chunk_size
        with th.no_grad():
            encoder_kv = None
            x = th.randn(bs, l, d).cuda()
            if self.attn_func == 6:
                encoder_kv = th.randn(bs, enc_l, d).cuda()

            self.del_cache()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=False)
            self.del_cache()
            y_forw_sample = self.forward(x, encoder_kv=encoder_kv, sample=True)
            max_err = th.max(th.abs(y_forw - y_forw_sample))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(y_forw - y_forw_sample)[:, i, :]) > 1e-6]}"

            self.del_cache()
            x_chunks = th.chunk(x, n_chunks, dim=1)
            y_chunks = []
            total_len = 0
            for x_chunk in x_chunks:
                y_chunk = self.forward(
                    x_chunk.contiguous(), encoder_kv=encoder_kv, sample=True
                )
                total_len += x_chunk.shape[1]
                self.check_cache(bs, total_len, False)
                y_chunks.append(y_chunk)
            y_forw_in_chunks = th.cat(y_chunks, dim=1)

            max_err = th.max(th.abs(y_forw - y_forw_in_chunks))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


class MLP(nn.Module):
    def __init__(
        self,
        n_in,
        n_state,
        resid_dropout=0.0,
        afn='quick_gelu',
        zero_out=False,
        init_scale=1.0,
    ):
        super().__init__()
        self.c_fc = Conv1D(n_in, n_state, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.act = ACT_FNS[afn]
        self.resid_dropout = (
            nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x
        )

    def forward(self, x):
        m = self.act(self.c_fc(x))
        m = self.c_proj(m)
        return self.resid_dropout(m)


class ResAttnBlock(nn.Module):
    def __init__(
        self,
        n_in,
        n_ctx,
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn='quick_gelu',
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        res_scale=1.0,
        m_attn=0.25,
        m_mlp=1.0,
        checkpoint_attn=0,
        checkpoint_mlp=0,
        attn_func=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.attn = FactoredAttention(
            n_in=n_in,
            n_ctx=n_ctx,
            n_state=int(m_attn * n_in),
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            scale=scale,
            mask=mask,
            zero_out=zero_out,
            init_scale=init_scale,
            checkpoint_attn=checkpoint_attn,
            attn_func=attn_func,
            blocks=blocks,
            spread=spread,
            encoder_dims=encoder_dims,
            prime_len=prime_len,
        )
        self.ln_0 = LayerNorm(n_in)
        self.mlp = MLP(
            n_in=n_in,
            n_state=int(m_mlp * n_in),
            resid_dropout=resid_dropout,
            afn=afn,
            zero_out=zero_out,
            init_scale=init_scale,
        )
        self.ln_1 = LayerNorm(n_in)
        self.res_scale = res_scale

        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_mlp = checkpoint_mlp
        self.n_in = n_in
        self.attn_func = attn_func

    def forward(self, x, encoder_kv, sample=False):
        if sample:
            a = self.attn(self.ln_0(x), encoder_kv, sample)
            m = self.mlp(self.ln_1(x + a))
        else:
            if self.attn_func == 6:
                assert encoder_kv is not None
                a = checkpoint(
                    lambda _x, _enc_kv, _s=sample: self.attn(
                        self.ln_0(_x), _enc_kv, _s
                    ),
                    (x, encoder_kv),
                    (*self.attn.parameters(), *self.ln_0.parameters()),
                    self.checkpoint_attn == 3,
                )  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            else:
                assert encoder_kv is None
                a = checkpoint(
                    lambda _x, _enc_kv=None, _s=sample: self.attn(
                        self.ln_0(_x), _enc_kv, _s
                    ),
                    (x,),
                    (*self.attn.parameters(), *self.ln_0.parameters()),
                    self.checkpoint_attn == 3,
                )  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            m = checkpoint(
                lambda _x: self.mlp(self.ln_1(_x)),
                (x + a,),
                (*self.mlp.parameters(), *self.ln_1.parameters()),
                self.checkpoint_mlp == 1,
            )
        if self.res_scale == 1.0:
            h = x + a + m
        else:
            h = x + self.res_scale * (a + m)
        return h


class Transformer(nn.Module):
    def __init__(
        self,
        n_in,
        n_ctx,
        n_head,
        n_depth,
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn='quick_gelu',
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        res_scale=False,
        m_attn=0.25,
        m_mlp=1.0,
        checkpoint_attn=0,
        checkpoint_mlp=0,
        checkpoint_res=0,
        attn_order=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.encoder_dims = encoder_dims
        self.blocks = blocks
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.prime_len = prime_len
        self.n_head = n_head

        res_scale = 1.0 / n_depth if res_scale else 1.0

        # Orders of attn_func
        attn_func = {
            0: lambda d: 0,  # Complete dense attn
            1: lambda d: [1, 2][d % 2],  # Alternate row and column attn
            2: lambda d: [1, 2, 3][
                d % 3
            ],  # Alternate row, column and previous row attn
            3: lambda d: [1, 4][d % 2],  # Alternate row and last column
            4: lambda d: [1, 5][d % 2],  # Alternate row and last k columns
            5: lambda d: [1, 4, 1, 1][
                d % 4
            ],  # Alternate row, last column, row, row
            6: lambda d: [1, 2, 3, 6][d % 4],
            7: lambda d: [*[1, 2, 3] * 5, 6][d % 16],
            8: lambda d: [1, 2, 3, 1, 2, 3, 1, 2, 3, 6][
                d % 10
            ],  # Used by separated_enc_dec model with lyrics
            9: lambda d: [1, 2, 3, 0][d % 4],
            10: lambda d: [
                *[1, 2, 3, 1, 2, 3, 1, 2, 3],
                *[1, 2, 3, 1, 2, 3, 1, 2, 3, 6] * 7,
            ][
                d % 79
            ],  # Used by large separated_enc_dec model with lyrics
            11: lambda d: [6, 6, 0][d % 3]
            if d % 16 == 15
            else [1, 2, 3][d % 3],
            12: lambda d: [7, 7, 0][d % 3]
            if d % 16 == 15
            else [1, 2, 3][d % 3],  # Used by single_enc_dec model with lyrics
        }[attn_order]

        attn_cycle = {
            0: 1,
            1: 2,
            2: 3,
            3: 2,
            4: 2,
            5: 4,
            6: 4,
            7: 16,
            8: 10,
            9: 4,
            10: 79,
            11: 16,
            12: 16,
        }[attn_order]
        # assert n_depth % attn_cycle == 0, f'Depth {n_depth} not a multiple of cycle {attn_cycle} for attn_order {attn_order}'

        attn_block = lambda d: ResAttnBlock(
            n_in=n_in,
            n_ctx=n_ctx,
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            afn=afn,
            scale=scale,
            mask=mask,
            zero_out=zero_out if attn_func(d) != 6 else True,
            init_scale=init_scale,
            res_scale=res_scale,
            m_attn=m_attn,
            m_mlp=m_mlp,
            checkpoint_attn=checkpoint_attn,
            checkpoint_mlp=checkpoint_mlp,
            attn_func=attn_func(d),
            blocks=blocks,
            spread=spread,
            encoder_dims=encoder_dims,
            prime_len=prime_len,
        )

        self.checkpoint_res = checkpoint_res
        self._attn_mods = nn.ModuleList()
        for d in range(n_depth):
            self._attn_mods.append(attn_block(d))
        self.ws = []

    def set_record_attn(self, record_attn):
        """
        Arguments:
            record_attn (bool or set): Makes forward prop dump self-attention
                softmaxes to self.ws. Either a set of layer indices indicating
                which layers to store, or a boolean value indicating whether to
                dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn

        for i, l in enumerate(self._attn_mods):
            l.attn.record_attn = _should_record_attn(i)
        if record_attn:
            assert self.ws == []
            for l in self._attn_mods:
                assert l.attn.w == None
        else:
            self.ws = []
            for l in self._attn_mods:
                l.attn.w = None

    def forward(
        self, x, encoder_kv=None, sample=False, fp16=False, fp16_out=False
    ):
        if fp16:
            x = x.half()

        # Blocks
        for i, l in enumerate(self._attn_mods):
            if self.checkpoint_res == 1 and not sample:
                if l.attn_func == 6:
                    assert encoder_kv is not None
                    f = functools.partial(l, sample=sample)
                    x = checkpoint(f, (x, encoder_kv), l.parameters(), True)
                else:
                    f = functools.partial(l, encoder_kv=None, sample=sample)
                    x = checkpoint(f, (x,), l.parameters(), True)
            else:
                if l.attn_func == 6:
                    x = l(x, encoder_kv=encoder_kv, sample=sample)
                else:
                    x = l(x, encoder_kv=None, sample=sample)
            if l.attn.record_attn:
                self.ws.append(l.attn.w)
        if not fp16_out:
            x = x.float()
        return x

    def check_cache(self, n_samples, sample_t, fp16):
        for l in self._attn_mods:
            l.attn.check_cache(n_samples, sample_t, fp16)

    def del_cache(self):
        for l in self._attn_mods:
            l.attn.del_cache()

    def check_sample(self):
        bs, l, s, d = (4, self.n_ctx, self.encoder_dims, self.n_in)
        prime = 5
        with th.no_grad():
            encoder_kv = th.randn(bs, s, d).cuda()
            x = th.randn(bs, l, d).cuda()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=True)

            self.del_cache()
            x_chunks = th.chunk(x, 4, dim=1)
            y_chunks = []
            n = 0
            for x_chunk in x_chunks:
                self.check_cache(bs, n, False)
                y_chunk = self.forward(
                    x_chunk, encoder_kv=encoder_kv, sample=True
                )
                y_chunks.append(y_chunk)
                n += x_chunk.shape[1]
            self.check_cache(bs, n, False)
            y_forw_in_chunks = th.cat(y_chunks, dim=1)

            max_err = th.max(th.abs(y_forw - y_forw_in_chunks))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer(
                'pos', th.tensor(get_pos_idx(input_shape)).long()
            )
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(
                get_normal(input_dims, width, std=0.01 * init_scale)
            )

    def forward(self):
        if self.pos_init:
            pos_emb = sum(
                [
                    self._pos_embs[i](self.pos[:, i])
                    for i in range(len(self.input_shape))
                ]
            )
        else:
            pos_emb = self.pos_emb
        return pos_emb


class ConditionalAutoregressive2D(nn.Module):
    def __init__(
        self,
        input_shape,
        bins,
        continuous=False,
        width=128,
        depth=2,
        heads=1,
        attn_dropout=0.0,
        resid_dropout=0.0,
        emb_dropout=0.0,
        mask=True,
        zero_out=False,
        init_scale=1.0,
        res_scale=False,
        pos_init=False,
        m_attn=0.25,
        m_mlp=1,
        checkpoint_res=0,
        checkpoint_attn=0,
        checkpoint_mlp=0,
        attn_order=0,
        blocks=None,
        spread=None,
        x_cond=False,
        y_cond=False,
        encoder_dims=0,
        only_encode=False,
        merged_decoder=False,
        prime_len=None,
        entropy_reg=0,
        tanh_output=False,
        mixture_size=1,
    ):
        super().__init__()
        self.continuous = continuous
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth
        self.entropy_reg = entropy_reg
        self.tanh_output = tanh_output

        if not self.continuous:
            self.x_emb = nn.Embedding(bins, width)
        else:
            self.x_emb = nn.Linear(bins, width, bias=False)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond or y_cond == 'frame':
            self.start_token = nn.Parameter(
                get_normal(1, width, std=0.01 * init_scale)
            )

        self.pos_emb = PositionEmbedding(
            input_shape=input_shape,
            width=width,
            init_scale=init_scale,
            pos_init=pos_init,
        )
        self.pos_emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            n_in=width,
            n_ctx=input_dims,
            n_head=heads,
            n_depth=depth,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            afn='quick_gelu',
            scale=True,
            mask=mask,
            zero_out=zero_out,
            init_scale=init_scale,
            res_scale=res_scale,
            m_attn=m_attn,
            m_mlp=m_mlp,
            checkpoint_attn=checkpoint_attn,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_res=checkpoint_res,
            attn_order=attn_order,
            blocks=blocks,
            spread=spread,
            encoder_dims=encoder_dims,
            prime_len=prime_len,
        )

        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            # Merged piped model uses this setup
            self.add_cond_after_transformer = False
            self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            if not self.continuous:
                self.share_x_emb_x_out = True
            else:
                self.share_x_emb_x_out = False

        self.mixture_size = mixture_size
        if not only_encode:
            if not self.continuous:
                self.x_out = nn.Linear(width, bins, bias=False)
            else:
                if mixture_size > 1:
                    self.x_out = nn.Linear(
                        width,
                        bins * 2 * mixture_size + mixture_size,
                        bias=False,
                    )
                else:
                    self.x_out = nn.Linear(width, bins * 2, bias=False)
                # self.x_out = nn.Linear(width, bins, bias=False)
            if self.share_x_emb_x_out:
                self.x_out.weight = self.x_emb.weight
            self.loss = th.nn.CrossEntropyLoss()

    def preprocess(self, x):
        # Input: x is NHWC and uint8. Converted to NL and long
        # Can include stuff like bitpacking, reordering here.
        N = x.shape[0]
        if self.continuous:
            return x.permute(0, 2, 1)  # NTC
        else:
            return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        # Convert back from NL and long to NHWC
        N = x.shape[0]
        if not self.continuous:
            assert (0 <= x).all() and (x < self.bins).all()
            if sample_tokens is None or sample_tokens == self.input_dims:
                return x.view(N, *self.input_shape)
            else:
                return x.view(N, -1)
        else:
            if sample_tokens is None or sample_tokens == self.input_dims:
                return x.view(N, self.input_shape[0], self.bins)
            else:
                return x.view(N, -1, self.bins)

    def forward(
        self,
        x,
        x_t,
        x_cond=None,
        y_cond=None,
        encoder_kv=None,
        fp16=False,
        loss_full=False,
        encode=False,
        get_preds=False,
        get_acts=False,
        get_sep_loss=False,
    ):
        # Preprocess.
        if isinstance(x_t, D.Normal):
            with th.no_grad():
                dist_t = D.Normal(
                    self.preprocess(x_t.loc), self.preprocess(x_t.scale)
                )
                x = self.preprocess(x)
        else:
            dist_t = None
            with th.no_grad():
                x = self.preprocess(x)
                x_t = self.preprocess(x_t)

        if self.continuous:
            N, C = x.shape[:2]
        else:
            N, C = x.shape
            assert isinstance(x, th.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()

        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width) or y_cond.shape == (
                N,
                C,
                self.width,
            )
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, C, self.width) or x_cond.shape == (
                N,
                1,
                self.width,
            ), f"{x_cond.shape} != {(N, C, self.width)} nor {(N, 1, self.width)}. Did you pass the correct --sample_length?"
        else:
            assert x_cond is None
            x_cond = th.zeros(
                (N, 1, self.width), device=x.device, dtype=th.float
            )

        # x_t = x  # Target
        x = self.x_emb(x)  # X emb
        x = roll(x, 1)  # Shift by 1, and fill in start token
        if self.y_cond == 'sequence':
            x[:, 0] = y_cond.view(N, self.width)
        else:
            x[:, 0] = self.start_token

        x = (
            self.x_emb_dropout(x)
            + self.pos_emb_dropout(self.pos_emb())
            + x_cond
        )  # Pos emb and dropout
        if self.y_cond == 'frame':
            x = x + y_cond

        x = self.transformer(x, encoder_kv=encoder_kv, fp16=fp16)  # Transformer
        if self.add_cond_after_transformer:  # Piped doesnt add x_cond
            x = x + x_cond
            if self.y_cond == 'frame':
                x = x + y_cond

        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x)  # Predictions
        if self.continuous:
            # if dist_t is not None:
            if self.mixture_size > 1:
                mu, std, mix = th.tensor_split(
                    x,
                    (
                        self.bins * self.mixture_size,
                        2 * self.bins * self.mixture_size,
                    ),
                    -1,
                )
                mus = mu.view(mu.shape[:2] + (self.mixture_size, -1))
                stds = std.view(std.shape[:2] + (self.mixture_size, -1))
                dist = D.MixtureSameFamily(
                    D.Categorical(logits=mix),
                    D.Independent(D.Normal(mus, F.softplus(stds)), 1),
                )
            else:
                mu, std = x.chunk(2, -1)
                dist = D.Normal(mu, F.softplus(std))

        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, : self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len :].reshape(-1, self.bins)

            prime_loss = F.cross_entropy(
                x_prime, x_t[:, : self.prime_len].reshape(-1)
            ) / np.log(2.0)
            gen_loss = F.cross_entropy(
                x_gen, x_t[:, self.prime_len :].reshape(-1)
            ) / np.log(2.0)

            loss = (prime_loss, gen_loss)  # Note order! Prime is first
        else:
            if not self.continuous:
                loss = F.cross_entropy(
                    x.view(-1, self.bins), x_t.view(-1), reduction='none'
                ) / np.log(
                    2.0
                )  # Loss
            else:
                if dist_t is not None:
                    loss = D.kl.kl_divergence(dist, dist_t)
                elif self.tanh_output:
                    loss = -D.TransformedDistribution(
                        dist, D.TanhTransform()
                    ).log_prob(x_t)
                    if self.mixture_size == 1:
                        loss -= self.entropy_reg * dist.entropy()
                    else:
                        loss -= (
                            self.entropy_reg
                            * dist.log_prob(dist.sample()).mean()
                        )
                else:
                    loss = F.gaussian_nll_loss(
                        dist.loc, x_t, dist.scale**2, reduction='none'
                    )
                    if self.mixture_size == 1:
                        loss -= self.entropy_reg * dist.entropy()
                    else:
                        loss -= (
                            self.entropy_reg
                            * dist.log_prob(dist.sample()).mean()
                        )

        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, C = n_samples, self.input_dims
        if sample_t == 0:
            # Fill in start token
            device = next(self.parameters()).device
            x = th.empty((n_samples, 1, self.width), device=device)
            if self.y_cond == 'sequence':
                x[:, 0] = y_cond.view(N, self.width)
            else:
                x[:, 0] = self.start_token
        else:
            if not self.continuous:
                assert isinstance(x, th.cuda.LongTensor)
                assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape[1] > 1:
            cond = x_cond[:, sample_t : sample_t + 1, :]
        else:
            cond = x_cond
        if self.y_cond == 'frame':
            if y_cond.shape[1] > 1:
                cond = cond + y_cond[:, sample_t : sample_t + 1, :]
            else:
                cond = cond + y_cond

        x = (
            x + self.pos_emb()[sample_t : sample_t + 1] + cond
        )  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def sample(
        self,
        n_samples,
        x_cond=None,
        y_cond=None,
        encoder_kv=None,
        fp16=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        get_dists=False,
        sample_tokens=None,
        mixture_temp=None,
    ):
        assert self.training == False

        if sample_tokens is None:
            sample_tokens = self.input_dims
        N, ID = n_samples, sample_tokens
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, ID, self.width) or y_cond.shape == (
                N,
                1,
                self.width,
            ), f"Got {y_cond.shape}, expected ({N}, {ID}/{1}, {self.width})"
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            if (
                x_cond.shape[1] == 8
            ):  # XXX hotfix for when a single conditioning vector is usampled
                x_cond = x_cond.narrow(1, 0, 1)
            assert x_cond.shape == (N, ID, self.width) or x_cond.shape == (
                N,
                1,
                self.width,
            ), f"Got {x_cond.shape}, expected ({N}, {ID}/{1}, {self.width})"
        else:
            assert x_cond is None
            device = next(self.parameters()).device
            x_cond = th.zeros((N, 1, self.width), dtype=th.float, device=device)

        with th.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            if get_dists:
                dists = []
            for sample_t in get_range(range(0, sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(
                    x, encoder_kv=encoder_kv, sample=True, fp16=fp16
                )  # Transformer
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x)  # Predictions
                if get_preds:
                    preds.append(x.clone())
                # Adjust logits
                if not self.continuous:
                    x = x / temp
                    x = filter_logits(x, top_k=top_k, top_p=top_p)
                    x = th.distributions.Categorical(
                        logits=x
                    ).sample()  # Sample and replace x
                    assert x.shape == (n_samples, 1)
                else:
                    if self.mixture_size > 1:
                        mu, std, mix = th.tensor_split(
                            x,
                            (
                                self.bins * self.mixture_size,
                                2 * self.bins * self.mixture_size,
                            ),
                            -1,
                        )
                        mus = mu.view(mu.shape[:2] + (self.mixture_size, -1))
                        stds = std.view(std.shape[:2] + (self.mixture_size, -1))
                        mtemp = (
                            mixture_temp if mixture_temp is not None else temp
                        )
                        if temp <= 0.0:
                            if mtemp <= 0.0:
                                idx = mix.argmax(dim=-1).squeeze(1)
                            else:
                                idx = D.Categorical(logits=mix / mtemp).sample()
                            x = dim_select(mus.squeeze(1), 1, idx).unsqueeze(1)
                        else:
                            dist = D.MixtureSameFamily(
                                D.Categorical(logits=mix / mtemp),
                                D.Independent(
                                    D.Normal(mus, F.softplus(stds) * temp), 1
                                ),
                            )
                            x = dist.sample()
                        if self.tanh_output:
                            x = th.tanh(x)
                        if get_dists:
                            dists.append(
                                D.MixtureSameFamily(
                                    D.Categorical(logits=mix.clone()),
                                    D.Independent(
                                        D.Normal(
                                            mus.clone(),
                                            F.softplus(stds.clone()),
                                        ),
                                        1,
                                    ),
                                )
                            )
                    else:
                        mu, std = x.chunk(2, -1)
                        if temp <= 0.0:
                            x = mu
                            if self.tanh_output:
                                x = th.tanh(x)
                        else:
                            dist = D.Normal(mu, F.softplus(std) * temp)
                            x = dist.sample()
                            if self.tanh_output:
                                x = th.tanh(x)
                        if get_dists:
                            dists.append(
                                D.Normal(
                                    mu.squeeze(1).clone(),
                                    F.softplus(std).squeeze(1).clone(),
                                )
                            )
                xs.append(x.clone())

            del x
            self.transformer.del_cache()

            x = th.cat(xs, dim=1)
            if get_preds:
                preds = th.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        elif get_dists:
            return x, dists
        else:
            return x

    def primed_sample(
        self,
        n_samples,
        x,
        x_cond=None,
        y_cond=None,
        encoder_kv=None,
        fp16=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        get_preds=False,
        get_dists=False,
        chunk_size=None,
        sample_tokens=None,
        mixture_temp=None,
    ):
        assert self.training == False

        if sample_tokens is None:
            sample_tokens = self.input_dims
        # Preprocess.
        with th.no_grad():
            x = self.preprocess(x)
        if not self.continuous:
            assert isinstance(x, th.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()
        assert x.shape[0] == n_samples
        xs = th.split(x, 1, dim=1)
        xs = list(xs)
        assert len(xs) < sample_tokens

        N, C = n_samples, sample_tokens
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, C, self.width) or y_cond.shape == (
                N,
                1,
                self.width,
            ), f"Got {y_cond.shape}, expected ({N}, {C}/{1}, {self.width})"
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, C, self.width) or x_cond.shape == (
                N,
                1,
                self.width,
            ), f"Got {x_cond.shape}, expected ({N}, {C}/{1}, {self.width})"
        else:
            assert x_cond is None
            device = next(self.parameters()).device
            x_cond = th.zeros((N, 1, self.width), dtype=th.float, device=device)

        with th.no_grad():
            if get_preds:
                preds = []
            if get_dists:
                dists = []

            # Fill up key/value cache for past context by runing forward pass.
            # We do so in chunks instead of doing the whole past in one forward pass to reduce max memory usage.
            if chunk_size is None:
                chunk_size = len(xs)
            # assert len(xs) % chunk_size == 0, f'expected {len(xs)} to be divisible by {chunk_size}'
            chunk_sizes = split_chunks(len(xs), chunk_size)
            x_primes = []
            start = 0
            x = None
            for current_chunk_size in get_range(chunk_sizes):
                xs_prime, conds_prime = [], []
                for sample_t in range(start, start + current_chunk_size):
                    x_prime, cond_prime = self.get_emb(
                        sample_t, n_samples, x, x_cond, y_cond
                    )
                    x = xs[sample_t]
                    xs_prime.append(x_prime)
                    conds_prime.append(cond_prime)
                start = start + current_chunk_size

                x_prime, cond_prime = th.cat(xs_prime, dim=1), th.cat(
                    conds_prime, dim=1
                )
                assert x_prime.shape == (
                    n_samples,
                    current_chunk_size,
                    self.width,
                )
                assert cond_prime.shape == (
                    n_samples,
                    current_chunk_size,
                    self.width,
                )
                del xs_prime
                del conds_prime
                if not get_preds:
                    del cond_prime
                x_prime = self.transformer(
                    x_prime, encoder_kv=encoder_kv, sample=True, fp16=fp16
                )

                if get_preds:
                    if self.add_cond_after_transformer:
                        x_prime = x_prime + cond_prime
                    assert x_prime.shape == (
                        n_samples,
                        current_chunk_size,
                        self.width,
                    )
                    del cond_prime
                    x_primes.append(x_prime)
                else:
                    del x_prime

            if get_preds:
                x_prime = th.cat(x_primes, dim=1)
                assert x_prime.shape == (n_samples, len(xs), self.width)
                x_prime = self.x_out(x_prime)  # Predictions
                preds.append(x_prime)

            # empty_cache() # this is slow, our models are not that big yet
            self.transformer.check_cache(n_samples, len(xs), fp16)

            x = xs[-1]
            if not self.continuous:
                assert x.shape == (n_samples, 1)
            # empty_cache()
            for sample_t in get_range(range(len(xs), sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(
                    x, encoder_kv=encoder_kv, sample=True, fp16=fp16
                )  # Transformer
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x)  # Predictions
                if get_preds:
                    preds.append(x)
                # Adjust logits
                if not self.continuous:
                    x = x / temp
                    x = filter_logits(x, top_k=top_k, top_p=top_p)
                    x = th.distributions.Categorical(
                        logits=x
                    ).sample()  # Sample and replace x
                    assert x.shape == (n_samples, 1)
                else:
                    if self.mixture_size > 1:
                        mu, std, mix = th.tensor_split(
                            x,
                            (
                                self.bins * self.mixture_size,
                                2 * self.bins * self.mixture_size,
                            ),
                            -1,
                        )
                        mus = mu.view(mu.shape[:2] + (self.mixture_size, -1))
                        stds = std.view(std.shape[:2] + (self.mixture_size, -1))
                        mtemp = (
                            mixture_temp if mixture_temp is not None else temp
                        )
                        if temp <= 0.0:
                            if mtemp <= 0.0:
                                idx = mix.argmax(dim=-1).squeeze(1)
                            else:
                                idx = D.Categorical(logits=mix / mtemp).sample()
                            x = dim_select(mus.squeeze(1), 1, idx).unsqueeze(1)
                        else:
                            dist = D.MixtureSameFamily(
                                D.Categorical(logits=mix / mtemp),
                                D.Independent(
                                    D.Normal(mus, F.softplus(stds) * temp), 1
                                ),
                            )
                            x = dist.sample()
                        if self.tanh_output:
                            x = th.tanh(x)
                        if get_dists:
                            dists.append(
                                D.MixtureSameFamily(
                                    D.Categorical(logits=mix.clone()),
                                    D.Independent(
                                        D.Normal(
                                            mus.clone(),
                                            F.softplus(stds.clone()),
                                        ),
                                        1,
                                    ),
                                )
                            )
                    else:
                        mu, std = x.chunk(2, -1)
                        if temp <= 0.0:
                            x = mu
                            if self.tanh_output:
                                x = th.tanh(x)
                        else:
                            dist = D.Normal(mu, F.softplus(std) * temp)
                            x = dist.sample()
                            if self.tanh_output:
                                x = th.tanh(x)
                        if get_dists:
                            dists.append(
                                D.Normal(
                                    mu.squeeze(1).clone(),
                                    F.softplus(std).squeeze(1).clone(),
                                )
                            )
                xs.append(x.clone())

            del x
            self.transformer.del_cache()

            x = th.cat(xs, dim=1)
            if get_preds:
                preds = th.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        elif get_dists:
            return x, dists
        else:
            return x

    def check_sample(self, chunk_size):
        bs, l, d = (4, self.input_dims, self.width)
        prime = int(self.input_dims // 8 * 7)
        enc_l = self.encoder_dims
        with th.no_grad():
            y_cond = th.randn(bs, 1, d).cuda() if self.y_cond else None
            x_cond = th.randn(bs, l, d).cuda() if self.x_cond else None
            encoder_kv = th.randn(bs, enc_l, d).cuda()

            x, preds_sample = self.sample(
                bs, x_cond, y_cond, encoder_kv, get_preds=True
            )
            loss, preds_forw = self.forward(
                x, x_cond, y_cond, encoder_kv, get_preds=True
            )
            max_err = th.max(th.abs(preds_sample - preds_forw))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            x_prime = x.view(bs, -1)[:, :prime]
            # unchunked
            x, preds_sample = self.primed_sample(
                bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True
            )
            assert (
                x.view(bs, -1)[:, :prime] == x_prime
            ).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(
                x, x_cond, y_cond, encoder_kv, get_preds=True
            )
            max_err = th.max(th.abs(preds_sample - preds_forw))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            # chunked
            x, preds_sample = self.primed_sample(
                bs,
                x_prime.clone(),
                x_cond,
                y_cond,
                encoder_kv,
                get_preds=True,
                chunk_size=chunk_size,
            )
            assert (
                x.view(bs, -1)[:, :prime] == x_prime
            ).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(
                x, x_cond, y_cond, encoder_kv, get_preds=True
            )
            max_err = th.max(th.abs(preds_sample - preds_forw))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if th.max(th.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"


class SimpleEmbedding(nn.Module):
    def __init__(self, bins, out_width, init_scale):
        super().__init__()
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)

    def forward(self, y):
        assert isinstance(
            y, th.cuda.LongTensor
        ), f"Expected dtype {th.cuda.LongTensor}, got {y.dtype}"
        assert (0 <= y).all() and (
            y < self.bins
        ).all(), f"Bins {self.bins}, got label {y}"
        if len(y.shape) == 3:
            # Sum up embeddings (bag-of-words) in case we get multiple labels
            return F.embedding_bag(
                y.view(y.shape[0] * y.shape[1], -1),
                self.emb.weight,
                padding_idx=0,
            ).view(y.shape[0], y.shape[1], -1)
        elif len(y.shape) == 2:
            return self.emb(y)
        else:
            raise ValueError('Expected shape with 2 or 3 dims, got {y.shape}')


class LabelConditioner(nn.Module):
    def __init__(self, y_bins, resolution, out_width, init_scale):
        super().__init__()
        self.resolution = resolution
        self.out_width = out_width
        self.emb = SimpleEmbedding(y_bins, out_width, init_scale)

    def forward(self, y):
        if self.resolution == 'sequence':
            return self.emb(y.narrow(1, 0, 1))
        elif self.resolution == 'frame':
            return self.emb(y)
        raise NotImplementedError(f'Unknown label resolution {self.resolution}')

        assert len(y.shape) == 2, f"Expected shape with 2 dims, got {y.shape}"
        assert (
            y.shape[-1] == 4 + self.max_bow_genre_size
        ), f"Expected shape (N,{4 + self.max_bow_genre_size}), got {y.shape}"
        assert isinstance(
            y, t.cuda.LongTensor
        ), f"Expected dtype {t.cuda.LongTensor}, got {y.dtype}"
        N = y.shape[0]
        total_length, offset, length, artist, genre = (
            y[:, 0:1],
            y[:, 1:2],
            y[:, 2:3],
            y[:, 3:4],
            y[:, 4:],
        )

        # Start embedding of length 1
        artist_emb = self.artist_emb(artist)
        # Empty genre slots are denoted by -1. We mask these out.
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(
            dim=1, keepdim=True
        )
        start_emb = genre_emb + artist_emb
        assert_shape(start_emb, (N, 1, self.out_width))

        # Pos embedding of length n_ctx
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length, start, end = (
                total_length.float(),
                start.float(),
                end.float(),
            )
            pos_emb = (
                self.total_length_emb(total_length)
                + self.absolute_pos_emb(start, end)
                + self.relative_pos_emb(
                    start / total_length, end / total_length
                )
            )
            assert_shape(pos_emb, (N, self.n_time, self.out_width))
        else:
            pos_emb = None
        return start_emb, pos_emb


class SimplePrior(nn.Module):
    def __init__(
        self,
        z_shapes,
        l_bins,
        continuous,
        encoder,
        decoder,
        level,
        downs_t,
        strides_t,
        labels,
        conditioner,
        prior_kwargs,
        x_cond_kwargs,
        y_cond_kwargs,
        prime_kwargs,
        copy_input,
        labels_v3=False,
        merged_decoder=False,
        single_enc_dec=False,
        deterministic_z=True,
        deterministic_tgt=True,
        deterministic_cond=True,
        bypass_encoder=False,
    ):
        super().__init__()

        self.use_tokens = prime_kwargs.pop('use_tokens')
        self.n_tokens = prime_kwargs.pop('n_tokens')
        self.prime_loss_fraction = prime_kwargs.pop('prime_loss_fraction')

        self.copy_input = copy_input
        if self.copy_input:
            prime_kwargs['bins'] = l_bins

        self.z_shapes = z_shapes
        self.levels = len(self.z_shapes)

        self.z_shape = self.z_shapes[level]

        self.level = level
        assert (
            level < self.levels
        ), f"Total levels {self.levels}, got level {level}"

        self.l_bins = l_bins

        # Passing functions instead of the vae module to avoid getting params
        self.encoder = encoder
        self.decoder = decoder
        self.deterministic_z = deterministic_z
        self.deterministic_tgt = deterministic_tgt
        self.deterministic_cond = deterministic_cond
        self.bypass_encoder = bypass_encoder

        # X conditioning
        self.x_cond = level != (self.levels - 1)
        self.cond_level = level + 1

        # Y conditioning
        self.y_cond = labels

        self.single_enc_dec = single_enc_dec
        # X conditioning
        if self.x_cond:
            self.conditioner_blocks = nn.ModuleList()
            cond_cls = Conditioner
            if conditioner == 'simple':
                cond_cls = SimpleConditioner
            conditioner_block = lambda _level: cond_cls(
                input_shape=z_shapes[_level],
                continuous=continuous,
                bins=l_bins,
                down_t=downs_t[_level],
                stride_t=strides_t[_level],
                **x_cond_kwargs,
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                log.debug(f"Conditioning on 1 above level(s)")
            self.conditioner_blocks.append(conditioner_block(self.cond_level))

        # Y conditioning
        if self.y_cond:
            self.y_emb = LabelConditioner(**y_cond_kwargs)

        # Lyric conditioning
        if single_enc_dec:
            # Single encoder-decoder transformer
            self.prior_shapes = [
                (self.n_tokens,),
                prior_kwargs.pop('input_shape'),
            ]
            self.prior_bins = [prime_kwargs['bins'], prior_kwargs.pop('bins')]
            self.prior_dims = [np.prod(shape) for shape in self.prior_shapes]
            self.prior_bins_shift = np.cumsum([0, *self.prior_bins])[:-1]
            self.prior_width = prior_kwargs['width']
            print_once(
                f'Creating cond. autoregress with prior bins {self.prior_bins}, '
            )
            print_once(f'dims {self.prior_dims}, ')
            print_once(f'shift {self.prior_bins_shift}')
            print_once(f'input shape {sum(self.prior_dims)}')
            print_once(f'input bins {sum(self.prior_bins)}')
            print_once(f'Self copy is {self.copy_input}')

            self.prime_loss_dims, self.gen_loss_dims = (
                self.prior_dims[0],
                self.prior_dims[1],
            )
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(
                input_shape=(sum(self.prior_dims),),
                bins=sum(self.prior_bins),
                # x_cond=(self.x_cond or self.y_cond),
                x_cond=self.x_cond,
                y_cond=True,
                prime_len=self.prime_loss_dims,
                **prior_kwargs,
            )

        else:
            # Separate encoder-decoder transformer
            if self.n_tokens != 0 and self.use_tokens:
                from jukebox.transformer.ops import Conv1D

                prime_input_shape = (self.n_tokens,)
                self.prime_loss_dims = np.prod(prime_input_shape)
                self.prime_acts_width, self.prime_state_width = (
                    prime_kwargs['width'],
                    prior_kwargs['width'],
                )
                self.prime_prior = ConditionalAutoregressive2D(
                    input_shape=prime_input_shape,
                    x_cond=False,
                    y_cond=False,
                    only_encode=True,
                    **prime_kwargs,
                )
                self.prime_state_proj = Conv1D(
                    self.prime_acts_width,
                    self.prime_state_width,
                    init_scale=prime_kwargs['init_scale'],
                )
                self.prime_state_ln = LayerNorm(self.prime_state_width)
                self.prime_bins = prime_kwargs['bins']
                self.prime_x_out = nn.Linear(
                    self.prime_state_width, self.prime_bins, bias=False
                )
                nn.init.normal_(
                    self.prime_x_out.weight,
                    std=0.02 * prior_kwargs['init_scale'],
                )
            else:
                self.prime_loss_dims = 0
            self.gen_loss_dims = np.prod(self.z_shape)
            self.total_loss_dims = self.prime_loss_dims + self.gen_loss_dims
            self.prior = ConditionalAutoregressive2D(
                continuous=continuous,
                # x_cond=(self.x_cond or self.y_cond),
                x_cond=self.x_cond,
                y_cond=self.y_cond,
                encoder_dims=self.prime_loss_dims,
                merged_decoder=merged_decoder,
                **prior_kwargs,
            )

        self.n_ctx = self.gen_loss_dims
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.cond_downsample = (
            self.downsamples[level + 1] if level != self.levels - 1 else None
        )
        self.raw_to_tokens = np.prod(self.downsamples[: level + 1])
        self.sample_length = self.n_ctx * self.raw_to_tokens
        if labels and False:  # no fancy labeling here
            self.labels_v3 = labels_v3
            self.labeller = Labeller(
                self.y_emb.max_bow_genre_size,
                self.n_tokens,
                self.sample_length,
                v3=self.labels_v3,
            )
        else:
            self.labeller = EmptyLabeller()

        log.debug(
            f"Level:{level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}"
        )

    def get_y(self, labels, start, get_indices=False):
        if isinstance(self.labeller, EmptyLabeller):
            return None
        y = labels['y'].clone()

        # Set sample_length to match this level
        y[:, 2] = int(self.sample_length)

        # Set offset
        y[:, 1:2] = y[:, 1:2] + int(start * self.raw_to_tokens)

        # Set lyric tokens
        indices = self.labeller.set_y_lyric_tokens(y, labels)
        if get_indices:
            return y, indices
        else:
            return y

    def get_z_conds(self, zs, start, end):
        if self.level != self.levels - 1:
            assert (
                start % self.cond_downsample == end % self.cond_downsample == 0
            )
            z_cond = zs[self.level + 1][
                :, start // self.cond_downsample : end // self.cond_downsample
            ]
            assert z_cond.shape[1] == self.n_ctx // self.cond_downsample
            z_conds = [z_cond]
        else:
            z_conds = None
        return z_conds

    def prior_preprocess(self, xs, conds):
        N = xs[0].shape[0]
        for i in range(len(xs)):
            x, shape, dims = xs[i], self.prior_shapes[i], self.prior_dims[i]
            bins, bins_shift = int(self.prior_bins[i]), int(
                self.prior_bins_shift[i]
            )
            assert isinstance(x, th.cuda.LongTensor), x
            assert (0 <= x).all() and (x < bins).all()
            # assert_shape(x, (N, *shape))
            xs[i] = (xs[i] + bins_shift).view(N, -1)

        for i in range(len(conds)):
            cond, shape, dims = (
                conds[i],
                self.prior_shapes[i],
                self.prior_dims[i],
            )
            if cond is not None:
                assert_shape(cond, (N, dims, self.prior_width))
            else:
                conds[i] = th.zeros(
                    (N, dims, self.prior_width), dtype=th.float, device='cuda'
                )

        return th.cat(xs, dim=1), th.cat(conds, dim=1)

    def prior_postprocess(self, z):
        N = z.shape[0]
        dims = (self.prior_dims[0], z.shape[1] - self.prior_dims[0])
        # xs = list(t.split(z, self.prior_dims, dim=1))
        xs = list(th.split(z, dims, dim=1))

        for i in range(len(xs)):
            # x, shape, dims, bins, bins_shift = xs[i], self.prior_shapes[i], self.prior_dims[i], self.prior_bins[i], self.prior_bins_shift[i]
            # assert_shape(x, (N, dims))
            shape = self.prior_shapes[i]
            bins, bins_shift = int(self.prior_bins[i]), int(
                self.prior_bins_shift[i]
            )
            # xs[i] = (xs[i] - bins_shift).view(N, *shape) #view(N, -1, *shape[1:])
            xs[i] = (xs[i] - bins_shift).view(N, -1, *shape[1:])
            xs[i] = th.clamp(
                xs[i], min=0
            )  # If not masking loss, model may have generated lyric/midi tokens which are now shifted <0 by bin_shift
            rank = dist.get_rank() if dist.is_initialized() else 0
            assert (
                xs[i] < bins
            ).all(), f'rank: {rank}, bins: {bins}, dims {dims}, shape {shape}, prior_shape {self.prior_shapes}, bins_shift {bins_shift}, xs[i]: {xs[i]}'

        return xs[-1]

    def x_emb(self, z_conds):
        z_conds = z_conds[: self.cond_level - self.level]
        assert (
            len(z_conds)
            == len(self.conditioner_blocks)
            == self.cond_level - self.level
        ), f"Expected {len(z_conds)} == {len(self.conditioner_blocks)} == {self.cond_level} - {self.level}"
        x_cond = None
        for z_cond, conditioner_block in reversed(
            list(zip(z_conds, self.conditioner_blocks))
        ):
            x_cond = conditioner_block(z_cond, x_cond)
        return x_cond

    @th.no_grad()
    def encode(self, x, deterministic=None):
        # Get latents
        zs = self.encoder(
            x,
            deterministic=self.deterministic_z
            if deterministic is None
            else deterministic,
        ).permute(0, 2, 1)
        return [zs]

    @th.no_grad()
    def decode(self, zs):
        x_out = self.decoder(zs[0])
        return x_out

    def get_cond(self, z_conds, y):
        if y is not None:
            '''
            assert (
                y.shape[1] == 4 + self.y_emb.max_bow_genre_size + self.n_tokens
            ), f"Expected {4} + {self.y_emb.max_bow_genre_size} + {self.n_tokens}, got {y.shape[1]}"
            n_labels = y.shape[1] - self.n_tokens
            y, prime = y[:, :n_labels], y[:, n_labels:]
            '''
            prime = None
        else:
            y, prime = None, None
        y_pos = None
        y_cond = self.y_emb(y) if self.y_cond else None
        x_cond = self.x_emb(z_conds) if self.x_cond else y_pos
        return x_cond, y_cond, prime

    def sample(
        self,
        n_samples,
        z=None,
        z_conds=None,
        y=None,
        fp16=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        chunk_size=None,
        sample_tokens=None,
        get_dists=False,
        mixture_temp=None,
    ):
        N = n_samples
        if z is not None:
            assert (
                z.shape[0] == N
            ), f"Expected shape ({N},**), got shape {z.shape}"
        if y is not None:
            assert (
                y.shape[0] == N
            ), f"Expected shape ({N},**), got shape {y.shape}"
        if z_conds is not None:
            for z_cond in z_conds:
                assert (
                    z_cond.shape[0] == N
                ), f"Expected shape ({N},**), got shape {z_cond.shape}"

        no_past_context = z is None or z.shape[1] == 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            name = {True: 'Ancestral', False: 'Primed'}[no_past_context]
            log.debug(
                f"{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}"
            )

        with th.no_grad():
            # Currently x_cond only uses immediately above layer
            x_cond, y_cond, prime = self.get_cond(z_conds, y)
            if self.single_enc_dec:
                # assert chunk_size % self.prime_loss_dims == 0. TODO: Check if needed
                if no_past_context:
                    z, x_cond = self.prior_preprocess([prime], [None, x_cond])
                else:
                    z, x_cond = self.prior_preprocess(
                        [prime, z], [None, x_cond]
                    )
                if sample_tokens is not None:
                    sample_tokens += self.n_tokens
                z = self.prior.primed_sample(
                    n_samples,
                    z,
                    x_cond,
                    y_cond,
                    fp16=fp16,
                    temp=temp,
                    mixture_temp=mixture_temp,
                    top_k=top_k,
                    top_p=top_p,
                    chunk_size=chunk_size,
                    sample_tokens=sample_tokens,
                    get_dists=get_dists,
                )
                if get_dists:
                    z, dists = z
                z = self.prior_postprocess(z)
            else:
                encoder_kv = self.get_encoder_kv(prime, fp16=fp16, sample=True)
                if no_past_context:
                    z = self.prior.sample(
                        n_samples,
                        x_cond,
                        y_cond,
                        encoder_kv,
                        fp16=fp16,
                        temp=temp,
                        mixture_temp=mixture_temp,
                        top_k=top_k,
                        top_p=top_p,
                        sample_tokens=sample_tokens,
                        get_dists=get_dists,
                    )
                else:
                    z = self.prior.primed_sample(
                        n_samples,
                        z,
                        x_cond,
                        y_cond,
                        encoder_kv,
                        fp16=fp16,
                        temp=temp,
                        mixture_temp=mixture_temp,
                        top_k=top_k,
                        top_p=top_p,
                        chunk_size=chunk_size,
                        sample_tokens=sample_tokens,
                        get_dists=get_dists,
                    )
                if get_dists:
                    z, dists = z
            if sample_tokens is None:
                if not self.prior.continuous:
                    assert_shape(z, (N, *self.z_shape))
                else:
                    assert_shape(z, (N, self.z_shape[0], self.prior.bins))
        if get_dists:
            return z, dists
        else:
            return z

    def get_encoder_kv(self, prime, fp16=False, sample=False):
        if self.n_tokens != 0 and self.use_tokens:
            if sample:
                self.prime_prior.cuda()
            N = prime.shape[0]
            prime_acts = self.prime_prior(prime, None, None, None, fp16=fp16)
            assert_shape(
                prime_acts, (N, self.prime_loss_dims, self.prime_acts_width)
            )
            assert (
                prime_acts.dtype == th.float
            ), f'Expected th.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert (
                encoder_kv.dtype == th.float
            ), f'Expected th.float, got {encoder_kv.dtype}'
            if sample:
                self.prime_prior.cpu()
                if fp16:
                    encoder_kv = encoder_kv.half()
        else:
            encoder_kv = None
        return encoder_kv

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(
                encoder_kv.view(-1, self.prime_bins), prime_t.view(-1)
            ) / np.log(2.0)
        else:
            prime_loss = th.tensor(0.0, device='cuda')
        return prime_loss

    def z_forward(
        self,
        z,
        z_conds=[],
        y=None,
        fp16=False,
        get_preds=False,
        get_attn_weights=False,
        z_smp=None,
        z_conds_smp=None,
    ):
        """
        Arguments:
            get_attn_weights (bool or set): Makes forward prop dump
                self-attention softmaxes to self.prior.transformer.ws. Either a
                set of layer indices indicating which layers to store, or a
                boolean value indicating whether to dump all.
        """
        assert isinstance(get_attn_weights, (bool, set))
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        if not self.deterministic_cond:
            x_cond, y_cond, prime = self.get_cond(z_conds_smp, y)
        else:
            x_cond, y_cond, prime = self.get_cond(z_conds, y)
        if self.copy_input:
            prime = z[:, : self.n_tokens]
        if self.single_enc_dec:
            raise NotImplementedError()  # deterministic/sampling z
            z, x_cond = self.prior_preprocess([prime, z], [None, x_cond])
            (prime_loss, gen_loss), preds = self.prior(
                z,
                x_cond,
                y_cond,
                fp16=fp16,
                get_sep_loss=True,
                get_preds=get_preds,
            )
        else:
            encoder_kv = self.get_encoder_kv(prime, fp16=fp16)
            prime_loss = self.get_prime_loss(encoder_kv, prime)
            if isinstance(z, D.Normal):
                z_inp = z.mean if self.deterministic_z else z.sample()
                z_tgt = z
            else:
                z_inp = z if self.deterministic_z else z_smp
                if self.training:
                    z_tgt = z if self.deterministic_tgt else z_smp
                else:
                    z_tgt = z
            gen_loss, preds = self.prior(
                z_inp,
                z_tgt,
                x_cond,
                y_cond,
                encoder_kv,
                fp16=fp16,
                get_preds=get_preds,
            )
        loss = (
            self.prime_loss_fraction
            * prime_loss
            * self.prime_loss_dims
            / self.total_loss_dims
        ) + (gen_loss * self.gen_loss_dims / self.total_loss_dims)
        metrics = dict(
            bpd=gen_loss.clone().detach(),
            prime_loss=prime_loss.clone().detach(),
            gen_loss=gen_loss.clone().detach(),
        )
        if get_preds:
            metrics["preds"] = preds.clone().detach()
        if get_attn_weights:
            ws = self.prior.transformer.ws
            self.prior.transformer.set_record_attn(False)
            return ws
        else:
            return loss, metrics

    def forward(self, x, y=None, fp16=False, decode=False, get_preds=False):
        if self.bypass_encoder:
            if x.shape[-1] == self.l_bins:
                eps = 1e-7
                z = x.permute(0, 2, 1).clamp(-1 + eps, 1 - eps)
                z_smp = z
                z_conds, z_conds_smp = [], []
            else:
                zmu, zstd = x.chunk(2, -1)
                z = zmu.permute(0, 2, 1)
                z_smp = (
                    D.Normal(zmu, F.softplus(zstd)).sample().permute(0, 2, 1)
                )
                z_conds, z_conds_smp = [], []
                if self.vae.tanh_latents:  # XXX
                    z = th.tanh(z)
                    z_smp = th.tanh(z_smp)
        else:
            if (
                len(
                    set(
                        [
                            self.deterministic_z,
                            self.deterministic_tgt,
                            self.deterministic_cond,
                        ]
                    )
                )
                > 1
            ):
                z, *z_conds = self.encode(x, deterministic=True)
                z_smp, *z_conds_smp = self.encode(x, deterministic=False)
            else:
                z, *z_conds = self.encode(x)
                z_smp, z_conds_smp = z, z_conds
        loss, metrics = self.z_forward(
            z=z,
            z_conds=z_conds,
            y=y,
            fp16=fp16,
            get_preds=get_preds,
            z_smp=z_smp,
            z_conds_smp=z_conds_smp,
        )
        if decode:
            x_out = self.decode([z, *z_conds])
        else:
            x_out = None
        return x_out, loss, metrics
