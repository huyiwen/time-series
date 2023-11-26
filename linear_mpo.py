# -*- coding: utf-8 -*-
import logging
import os
import re
from typing import Optional, Tuple, List, Set
import math
from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch import nn
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init


os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = logging.getLogger(__name__)


def state_dict_matrix_to_mpo(state_dict: dict, target_parameters: Set[str], mapping, return_inflate_rate=False):
    new_state_dict = {}
    if isinstance(target_parameters, dict):
        target_parameters = set(target_parameters.keys())
    # print(target_parameters)
    inflate_rate = 0
    mpo_num = 0
    truncate_num = 10000000
    optimizer_grouped_parameters = {}
    for k, v in state_dict.items():
        if k not in target_parameters and k.endswith('.weight'):

            if "word_embeddings" in k:
                input_output_shapes = mapping[v.shape[0]], mapping[v.shape[1]]
                mpo = EmbeddingMPO(*v.shape, *input_output_shapes, truncate_num, _weight=v)
                tensor_set = mpo.tensor_set
            else:
                input_output_shapes = mapping[v.shape[1]], mapping[v.shape[0]]
                mpo = LinearDecomMPO(*v.T.shape, *input_output_shapes, truncate_num, _weight=v)
                tensor_set = mpo.tensor_set

            param_num = 0
            for i, tensor in enumerate(tensor_set):
                name = k.rsplit(".", 1)[-2] + ".tensor_set." + str(i)
                # print(name, array, type(array))
                new_state_dict[name] = tensor
                param_num += np.prod(tensor.shape)

            cur_inflate = param_num / np.prod(v.shape)
            if cur_inflate not in optimizer_grouped_parameters:
                optimizer_grouped_parameters[cur_inflate] = []

            for i, tensor in enumerate(tensor_set):
                name = k.rsplit(".", 1)[-2] + ".tensor_set." + str(i)
                optimizer_grouped_parameters[cur_inflate].append(name)

            # inflate_rate += cur_inflate
            # mpo_num += 1
            print(f"{k:51}", *input_output_shapes, *v.shape, f"{cur_inflate:.2f}")
            # print(k, *input_output_shapes, v.shape, param_num / np.prod(v.shape), mpo.mpo.test_difference(tensor_set, v))
        else:
            new_state_dict[k] = v

    if return_inflate_rate:
        return new_state_dict, optimizer_grouped_parameters
    else:
        return new_state_dict


def state_dict_mpo_to_matrix(state_dict: dict) -> dict:
    new_state_dict = {}
    tensor_sets = defaultdict(dict)
    for k, v in state_dict.items():
        suffix = re.search(r".*(?=\.tensor_set\.\d+$)", k)
        if suffix is not None:
            idx = int(k.rsplit('.', 1)[-1])
            weight_name = suffix[0] + ".weight"
            tensor_sets[weight_name][idx] = v
        else:
            new_state_dict[k] = v
    for weight_name, tensor_set in tensor_sets.items():
        sorted_tensor_set = list(zip(*sorted(tensor_set.items(), key=lambda x: x[0])))[1]
        new_state_dict[weight_name] = MPO.self_adapted_mpo2matrix(sorted_tensor_set)
    return new_state_dict


class MPO:

    def __init__(self, mpo_input_shape: List[int], mpo_output_shape: List[int], truncate_num: int, fix_rank=None):
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.truncate_num = truncate_num
        self.num_dim = len(mpo_input_shape)
        self.mpo_ranks = self.compute_rank(truncate_num=None)
        if fix_rank:
            self.mpo_truncate_ranks = fix_rank
        else:
            self.mpo_truncate_ranks = self.compute_rank(truncate_num=self.truncate_num)

    @staticmethod
    def tensor_set_to_shapes(tensor_set) -> Tuple[List[int], List[int]]:
        input_shape = []
        output_shape = []
        for tensor in tensor_set:
            i, o = tensor.shape[1:-1]
            input_shape.append(i)
            output_shape.append(o)
        return input_shape, output_shape

    @staticmethod
    def self_adapted_mpo2matrix(tensor_set):
        input_shape, output_shape = MPO.tensor_set_to_shapes(tensor_set)
        return MPO(input_shape, output_shape, 10000).mpo2matrix(tensor_set)

    def compute_rank_position(self, s, truncate_num=None):

        """
        Calculate the rank position in MPO bond dimension
        :param s: target bond ,type = int, range in [1:len(mpo_input_shape-1)], r_0 = r_n = 1.
        :return:  target bond 's' real bond dimension.
        """
        rank_left = 1  # ranks_left: all the shape multiply in left of 's'.
        rank_right = 1  # ranks_right: all the shape multiply in right of 's'.
        for i in range(0, s):
            rank_left = rank_left * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        for i in range(s, self.num_dim):
            rank_right = rank_right * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        if truncate_num == None:
            min_rank = min(rank_left, rank_right)
        else:
            min_rank = min(int(self.truncate_num), rank_left, rank_right)
        return min_rank

    def compute_rank(self, truncate_num):
        """
        :param mpo_input_shape: the input mpo shape, type = list. [i0,i1,i2,...,i_(n-1)]
        :param truncate_num: the truncate number of mpo, type = int.
        :return:max bond dimension in every bond position, type = list, [r0,r1,r2,...,r_n],r0=r_n=1
        """
        bond_dims = [1 for i in range(self.num_dim + 1)]
        for i in range(1, self.num_dim):
            bond_dims[i] = self.compute_rank_position(i, truncate_num)
        return bond_dims

    def get_tensor_set(self, inp_matrix: NDArray):
        """
        Calculate the left canonical of input matrix with a given mpo_input_shape
        :param inp_matrix: the input matrix
        :param mpo_input_shape:
        :return: a tensor with left canonical in input matrix
        """
        # TODO: translate to pytorch version
        tensor_set = []
        res = inp_matrix
        #################################################################################

        res = res.reshape(tuple(self.mpo_input_shape[:]) + tuple(self.mpo_output_shape[:]))
        self.index_permute = np.transpose(
            np.array(range(len(self.mpo_input_shape) + len(self.mpo_output_shape))).reshape((2, -1))).flatten()
        res = np.transpose(res, self.index_permute)
        #################################################################################
        for i in range(self.num_dim - 1):
            # Do the SVD operator
            res = res.reshape([self.mpo_ranks[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i], -1])
            u, lamda, v = np.linalg.svd(res, full_matrices=False)
            # The first tensor should be T1(r_i+1, m_i, n_i, r_i)
            u = u.reshape([self.mpo_ranks[i], self.mpo_input_shape[i], self.mpo_output_shape[i], self.mpo_ranks[i+1]])
            tensor_set.append(u)
            res = np.dot(np.diag(lamda), v)
        res = res.reshape([self.mpo_ranks[self.num_dim-1], self.mpo_input_shape[self.num_dim-1],
                           self.mpo_output_shape[self.num_dim-1], self.mpo_ranks[self.num_dim]])
        tensor_set.append(res)
        return tensor_set

    def empty2mpo(self, device=None, dtype=None, requires_grad=True):
        shapes = []
        for i in range(self.num_dim):
            shapes.append((
                self.mpo_ranks[i],
                self.mpo_input_shape[i],
                self.mpo_output_shape[i],
                self.mpo_ranks[i+1]
            ))
        return nn.ParameterList([
            nn.Parameter(torch.empty(i, device=device, dtype=dtype, requires_grad=requires_grad)) for i in shapes
        ])

    def left_canonical(self,tensor_set):
        left_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        mat = tensor_set[0]
        mat = mat.reshape(-1, mat.shape[3])
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        left_canonical_tensor[1] = np.dot(np.diag(lamda), v)
        for i in range(1,self.num_dim-1):
            mat = np.tensordot(left_canonical_tensor[i], tensor_set[i],[1,0])
            mat = mat.reshape(-1, mat.shape[-1])
            u,lamda,v = np.linalg.svd(mat, full_matrices=False)
            left_canonical_tensor[i+1] = np.dot(np.diag(lamda), v)
        return left_canonical_tensor

    def right_canonical(self, tensor_set):
        """
        Calculate the right tensor canonical for MPO format required
        :param left_tensor: the tensor_set output from function: left_canonical
        :return: the right_tensor_canonical format for calculate the mpo decomposition
        """
        right_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        # print(tensor_set.shape)
        mat = tensor_set[self.num_dim - 1]
        mat = mat.reshape(mat.shape[0], -1)
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        right_canonical_tensor[self.num_dim - 1] = np.dot(u, np.diag(lamda))

        for i in range(self.num_dim - 2, 0, -1):
            mat = np.tensordot(tensor_set[i], right_canonical_tensor[i + 1], [3, 0])
            mat = mat.reshape(mat.shape[0], -1)
            u, lamda, v = np.linalg.svd(mat, full_matrices=False)
            right_canonical_tensor[i] = np.dot(u, np.diag(lamda))
        return right_canonical_tensor

    def expectrum_normalization(self, lamda):
        """
        Do the lamda normalization for calculate the needed rank for MPO structure
        :param lamda: lamda parameter from left canonical
        :return:
        """
        # print(lamda, lamda.shape)
        norm_para = np.sum(lamda ** 2) ** (0.5)
        lamda_n = lamda / norm_para
        lamda_12 = lamda ** (-0.5)
        return lamda_n, np.diag(lamda_12)

    def gauge_aux_p_q(self, left_canonical_tensor, right_canonical_tensor):
        p = [0 for i in range(self.num_dim + 1)]
        q = [0 for i in range(self.num_dim + 1)]
        lamda_set = [0 for i in range(self.num_dim + 1)]
        lamda_set_value = [0 for i in range(self.num_dim + 1)]
        lamda_set[0] = np.ones([1,1])
        lamda_set[-1] = np.ones([1,1])
        for i in range(1, self.num_dim):
            mat = np.dot(left_canonical_tensor[i],right_canonical_tensor[i])
            # mat = right_canonical_tensor[i]
            u, lamda, v = np.linalg.svd(mat)
            lamda_n, lamda_l2 = self.expectrum_normalization(lamda)
            lamda_set[i] = lamda_n
            lamda_set_value[i] = lamda
            p[i] = np.dot(right_canonical_tensor[i], v.T)
            p[i] = np.dot(p[i],lamda_l2)
            q[i] = np.dot(lamda_l2,u.T)
            q[i] = np.dot(q[i], left_canonical_tensor[i])
        return p, q, lamda_set, lamda_set_value

    def mpo_canonical(self, tensor_set, p, q):
        tensor_set[0] = np.tensordot(tensor_set[0], p[1], [3,0])
        tensor_set[-1] = np.tensordot(q[self.num_dim-1], tensor_set[-1], [1,0])
        for i in range(1, self.num_dim-1):
            tensor_set[i] = np.tensordot(q[i],tensor_set[i],[1,0])
            tensor_set[i] = np.tensordot(tensor_set[i],p[i+1], [3,0])
        return tensor_set

    def truncated_tensor(self, tensor_set, step_train=False):
        """
        Get a untruncated tensor by mpo
        :param tensor_set: the input weight
        :return: a untruncated tensor_set by mpo
        """
        if step_train:
            tensor_set_tmp = [i.detach().cpu().numpy() for i in tensor_set]
            cano_tensor_set = self.bi_canonical(tensor_set_tmp)
            tensor_set = torch.nn.ParameterList(
            [nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in cano_tensor_set])
            tensor_set[2].requires_grad = False

        mpo_trunc = self.mpo_truncate_ranks[:]
        for i in range(self.num_dim):
            if step_train:
                mask_noise = torch.ones_like(tensor_set[i])
            t = tensor_set[i]
            r_l = mpo_trunc[i]
            r_r = mpo_trunc[i + 1]
            if isinstance(tensor_set[i], nn.parameter.Parameter):
                if step_train:

                    mask_noise[r_l:, :, :, :] = 0.0
                    mask_noise[:r_l, :, :, r_r:] = 0.0
                    tensor_set[i].data = tensor_set[i].data * mask_noise
                else:
                    tensor_set[i].data = t[:r_l, :, :, :r_r]
            else:
                tensor_set[i] = t[:r_l, :, :, :r_r]
                assert "Check! tensor_set is not nn.parameter.Parameter"
        return tensor_set

    def matrix2mpo(self, inp_matrix, cutoff=True):
        """
        Utilize the matrix to mpo format with or without cutoff
        :param inp_matrix: the input matrix, type=list
        :param cutoff: weather cut of not, type = bool
        :return: the truncated of not mps format of input matrix
        """
        tensor_set = self.get_tensor_set(inp_matrix)
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,lamda_set, lamda_set_value = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)
        if cutoff != False:
            tensor_set = self.truncated_tensor(tensor_set)
        return tensor_set, lamda_set, lamda_set_value

    def bi_canonical(self, tensor_set):
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,_, _ = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)

        return tensor_set

    def mpo2matrix(self, tensor_set) -> Tensor:
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        # Squeeze the first and the last 1 dimension
        t = t.squeeze(0)
        t = t.squeeze(-1)
        # Caculate the new index for mpo
        tmp1 = torch.tensor(range(len(self.mpo_output_shape))) * 2
        tmp2 = tmp1 + 1
        new_index = torch.cat((tmp1, tmp2), 0)
        # Transpose and reshape to output
        t = t.permute(tuple(new_index))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        # TODO: use MPOTensor to avoid unexpected errors
        return t

    def calculate_total_mpo_param(self, cutoff=True):
        # print("use cutoff: ", cutoff)
        total_size = 0
        if cutoff:
            rank = self.mpo_truncate_ranks
        else:
            rank = self.mpo_ranks
        for i in range(len(self.mpo_input_shape)):
            total_size += rank[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i] * rank[i + 1]

        return total_size

    def new_mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t

    def test_difference(self, tensor_set, matrix):
        """
        we input an matrix , return the difference between those two matrix
        :param matrix:
        :return:
        """
        if isinstance(matrix, Tensor):
            matrix = matrix.detach().cpu().numpy()
        new_matrix = self.mpo2matrix(tensor_set).detach().cpu()

        transpose = None
        if new_matrix.shape[0] < matrix.shape[0] or new_matrix.shape[1] < matrix.shape[1]:
            # transpose = new_matrix
            new_matrix = new_matrix.reshape(matrix.shape)
        assert new_matrix.shape[0] >= matrix.shape[0] and new_matrix.shape[1] >= matrix.shape[1],\
            f"error matrix size {new_matrix.shape} and {matrix.shape}"

        v = new_matrix[:matrix.shape[0]].numpy() - matrix
        # error = np.linalg.norm(v)
        error = np.sum(v ** 2)
        if error > 1e-8:
            print(f"Warning: MPO decomposition has a large error {error}. Please consider set a larger truncate_num")
            print("truncate_num:", self.truncate_num)
            print("tensor_set\n", tensor_set)
            # print(f"transpose {transpose.shape}\n", transpose)
            print(f"new_matrix {new_matrix.shape}\n", new_matrix)
            print(f"matrix {matrix.shape}\n", matrix)
            print(f"norm:", np.linalg.norm(v))
            torch.save((tensor_set, matrix), "error_mpo_matrix.pth")
            exit()

        return error


class LinearDecomMPO_(nn.Module):
    '''
    compress using MPO method
    ref: Compressing deep neural networks by matrix product operators
    '''
    def __init__(
        self,
        input_size,
        output_size,
        mpo_input_shape,
        mpo_output_shape,
        truncate_num,
        bias = True,
        tensor_set=None,
        bias_tensor=None,
        device=None,
    ):
        super().__init__()
        self.tensor_set = None
        self.mpo = MPO(np.array(mpo_input_shape), np.array(mpo_output_shape), truncate_num)

        if tensor_set is not None:
            self._from_pretrained(tensor_set, bias_tensor, bias, device)
        else:
            # not register as parameter
            _linear = nn.Linear(input_size, output_size, bias=bias, device=device)
            self._from_pretrained(
                weight=_linear.weight.data.cpu().numpy(),
                bias_tensor=_linear.bias if bias else None,
                bias=bias,
                device=device
            )
            assert self.get_weight().shape == _linear.weight.data.shape

    def get_weight(self):
        return self.mpo.mpo2matrix(self.tensor_set).T

    def forward(self, x):
        res = x.reshape(-1, x.shape[-1])
        res = F.linear(res, self.get_weight(), self.bias_tensor)
        ori_shape=x.shape
        return res.view((tuple(ori_shape[:-1])+(-1,)))

    def _from_pretrained(self, weight: NDArray, bias_tensor: Optional[Tensor] = None, bias=True, device=None):
        tensor_set, _, _ = self.mpo.matrix2mpo(weight)
        # register tensor_set as parameter
        if device is not None:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).to(device)) for i in tensor_set])
        else:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i)) for i in tensor_set])

        if bias:
            self.bias_tensor = bias_tensor
        else:
            logger.info("Check no bias")
            self.bias_tensor = None


class MPOMixin:

    def _set_mpo(self, mpo_input_shape, mpo_output_shape, truncate_num, output="reshape", output_shape=None):
        self.mpo = MPO(np.array(mpo_input_shape), np.array(mpo_output_shape), truncate_num)
        self._output = output
        self._output_shape = output_shape
        if self._output == "reshape":
            assert isinstance(self._output_shape, tuple),\
                "reshape method should provide target shapes with a tuple"
        elif self._output == "slice":
            assert isinstance(self._output_shape, int),\
                "truncate to slice method should provide target slice number"

    def _parameter_decompose(self, weight: NDArray, device=None, dtype=None, requires_grad=True, mpo_initialize=True) -> nn.ParameterList:
        if mpo_initialize:
            tensor_set, _, _ = self.mpo.matrix2mpo(weight, cutoff=False)
            # register tensor_set as parameter
            return nn.ParameterList([
                nn.Parameter(torch.from_numpy(i).to(device=device, dtype=dtype), requires_grad=requires_grad) for i in tensor_set
            ])
        else:
            return self.mpo.empty2mpo(device=device, dtype=dtype, requires_grad=requires_grad)

    @property
    def weight(self) -> Tensor:
        if self._output == "reshape":
            return self.mpo.mpo2matrix(self.tensor_set).reshape(self._output_shape)
        elif self._output == "slice":
            return self.mpo.mpo2matrix(self.tensor_set)[:self._output_shape]
        else:
            return self.mpo.mpo2matrix(self.tensor_set)

    def _mpo_set_weight(self, weight):
        # print("weight", weight.shape)
        if isinstance(weight, Tensor):
            weight = weight.detach().cpu().numpy()
        self.tensor_set = self._parameter_decompose(weight)

    def _mpo_set_bias(self, bias):
        # print("bias", bias.shape)
        self.bias = bias


class LinearDecomMPO(nn.Module, MPOMixin):
    '''
    compress using MPO method
    ref: Compressing deep neural networks by matrix product operators
    '''

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    tensor_set: nn.ParameterList

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mpo_input_shape: List[int],
        mpo_output_shape: List[int],
        truncate_num: int = 1000000,
        bias: bool =True,
        _weight: Optional[Tensor] =None,
        bias_tensor: Optional[Tensor] = None,
        mpo_initialize: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._set_mpo(mpo_input_shape, mpo_output_shape, truncate_num, "reshape", (out_features, in_features))

        # not register as parameter
        if _weight is None:
            _weight = nn.Linear(in_features, out_features).weight.data.cpu().numpy()
        elif isinstance(_weight, Tensor):
            _weight = _weight.cpu().numpy()

        if bias:
            if bias_tensor is None:
                self.bias = nn.Parameter(torch.randn(out_features, **factory_kwargs))
            else:
                self.bias = nn.Parameter(bias_tensor)
        else:
            self.register_parameter('bias', None)

        self.tensor_set =  self._parameter_decompose(weight=_weight, mpo_initialize=mpo_initialize, **factory_kwargs)
        # print(self.tensor_set)
        # print(in_features, out_features, mpo_input_shape, mpo_output_shape, _weight.shape, self.tensor_set)
        self.last = None
        self.curr = None

    def forward(self, input: Tensor) -> Tensor:
        # print(input.dtype, self.weight.dtype, self.bias.dtype)
        # # print("With T", input.shape, self.weight.shape, self.bias.shape)
        # self.last = self.curr
        # self.curr = self.tensor_set[0]
        # if self.curr is not None and self.last is not None:
        #     pass
        #     # print(torch.sum((self.last - self.curr) ** 2))

        # res = input.reshape(-1, input.shape[-1])
        # res = F.linear(res, self.weight, self.bias)
        # ori_shape = input.shape
        # return res.view((tuple(ori_shape[:-1])+(-1,)))
        return F.linear(input, self.weight, self.bias)
        original_shape = input.shape
        flatten_input = input.reshape(-1, input.shape[-1])
        flatten_output = F.linear(flatten_input, self.weight, self.bias)
        return flatten_output.reshape(original_shape[:-1] + (-1,))


class EmbeddingMPO(nn.Module, MPOMixin):

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse', 'mpo']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    tensor_set: nn.ParameterList
    sparse: bool
    mpo: MPO

    def __init__(self, num_embeddings: int, embedding_dim: int, mpo_input_shape: List[int], mpo_output_shape: List[int],
                 truncate_num: int = 1000000, padding_idx: Optional[int] = None, max_norm: Optional[float] = None,
                 norm_type: float = 2., scale_grad_by_freq: bool = False, sparse: bool = False, freeze: bool = False,
                 _weight: Optional[Tensor] = None, device=None, dtype=None, mpo_initialize=True) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self._set_mpo(mpo_input_shape, mpo_output_shape, truncate_num, "slice", self.num_embeddings)

        mpo_input_prod = int(np.prod(mpo_input_shape).item())
        mpo_output_prod = int(np.prod(mpo_output_shape).item())

        if _weight is None:
            _weight = nn.Embedding(mpo_input_prod, mpo_output_prod).weight.data.cpu().numpy()
            # print("weight:", _weight.shape)
        else:
            if isinstance(_weight, Tensor):
                _weight = _weight.cpu().numpy()
            assert _weight.shape[1] == mpo_output_prod, \
                f'Embedding dim shoud have same siaze, but recieve {_weight.shape[1]} which is different from {mpo_output_prod}'
            extend_dim = (mpo_input_prod - _weight.shape[0], mpo_output_prod)
            _weight = np.concatenate([_weight, np.zeros(extend_dim)], axis=0)

        # replace weight with a set of tensors decomposed by MPO
        self.tensor_set = self._parameter_decompose(_weight, requires_grad=not freeze, device=device, dtype=dtype, mpo_initialize=mpo_initialize)
        self.sparse = sparse
        # print(self.weight.shape)

    def reset_parameters(self):
        pass

    def forward(self, input: Tensor) -> Tensor:
        # print(input.shape, self.weight.shape)
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
