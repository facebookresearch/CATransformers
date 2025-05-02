"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
"""
Implementation of the following modules is borrowed from from https://github.com/msr-fiddle/phaze
Licensed under MIT License
"""

import math
import copy
from ast import literal_eval
from math import prod
import op_to_compute


def get_problem_dim(schd, node):
    if (
        node.node_desc == "CudnnConvolution"
        or node.node_desc == "MkldnnConvolution"
        or node.node_desc == "fused ~~ MkldnnConvolution"
        or node.node_desc == "fused ~~ CudnnConvolution"
    ):
        output_act = node.output_act[0]
        input_act = node.saved_tensors[0]
        weight = node.saved_tensors[1]

        N = output_act[0]
        M = output_act[1]
        P = output_act[2]
        Q = output_act[3]

        W = input_act[2]
        H = input_act[3]

        C = weight[1]
        R = weight[2]
        S = weight[3]

        B = 1

        W_stride = node.stride[0]
        H_stride = node.stride[1]
        W_dilation = node.dilation[0]
        H_dilation = node.dilation[1]
        W_padding = node.padding[0]
        H_padding = node.padding[1]

        Type = "CONV"

        # Logic for Depth wise Convolution
        if input_act[1] != weight[1] and weight[1] == 1:

            N = output_act[0]
            M = 1
            C = output_act[1]
            P = output_act[2]
            Q = output_act[3]

            W = input_act[2]
            H = input_act[3]

            R = weight[2]
            S = weight[3]

            B = 0

            Type = "DSCONV"

            W_stride = node.stride[0]
            H_stride = node.stride[1]
            W_dilation = node.dilation[0]
            H_dilation = node.dilation[1]
            W_padding = node.padding[0]
            H_padding = node.padding[1]

        else:
            assert (
                output_act[0] == input_act[0]
                and output_act[1] == weight[0]
                and input_act[1] == weight[1]
            ), "Dimensions doesn't match!!"

    elif node.node_desc == "ThnnConv2D" or node.node_desc == "fused ~~ ThnnConv2D":
        output_act = node.output_act[0]
        input_act = node.saved_tensors[0]
        weight = node.saved_tensors[1]

        N = output_act[0]
        M = output_act[1]
        P = output_act[2]
        Q = output_act[3]

        W = input_act[2]
        H = input_act[3]

        C = weight[1]
        R = weight[2]
        S = weight[3]

        B = 1

        W_stride = node.stride[0]
        H_stride = node.stride[1]
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        Type = "CONV"

        assert (
            output_act[0] == input_act[0]
            and output_act[1] == weight[0]
            and input_act[1] == weight[1]
        ), "Dimensions doesn't match!!"

    elif node.node_desc == "Addmm" or node.node_desc == "fused ~~ Addmm" or node.node_desc == "fused ~~ Linear" or node.node_desc == "Linear":

        if len(node.saved_tensors) == 3:
            output_act = node.output_act[0]
            input_act = node.saved_tensors[1]
            weight = node.saved_tensors[2]

        elif len(node.saved_tensors) == 2:
            output_act = node.output_act[0]
            input_act = node.saved_tensors[0]
            weight = node.saved_tensors[1]

        elif len(node.saved_tensors) == 1:
            pred_nodes = schd.get_predecessor_nodes(node)

            """
            for pred_node in pred_nodes:
                print("predecessor : ", pred_node)
            
            for pred_node in pred_nodes:
                if pred_node.node_desc != 'AccumulateGrad' and pred_node.node_desc != 'T':
                    input_act = pred_node.output_act[0]
                    break
            """

            for pred_node in pred_nodes:
                if pred_node.node_desc == "T":
                    weight = pred_node.output_act[0]
                    break

            output_act = node.output_act[0]
            input_act = node.saved_tensors[0]

        weight = weight[-2:]
        output_act = output_act[-2:]

        if node.node_desc == "fused ~~ Linear" or node.node_desc == "Linear":
            # transposed dimensions
            weight = [weight[1], weight[0]]

        if(len(output_act) > 2):
            B = prod(output_act[:-2])
            assert output_act[0] == weight[0] == input_act[0], "Dimensions doesn't match!!"
        else:
            B = 1

        N = output_act[0]
        M = output_act[1]
        P = 1
        Q = 1

        W = 1
        H = 1

        C = weight[0]
        R = 1
        S = 1

        W_stride = 1
        H_stride = 1
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        Type = "CONV"

        assert output_act[1] == weight[1], "Dimensions doesn't match!!"

    elif node.node_desc == "Mm" or node.node_desc == "fused ~~ Mm":
        output_act = node.output_act[0]
        input_act = node.saved_tensors[0]
        weight = node.saved_tensors[1]

        B = 1

        if len(output_act) == 3:
            assert output_act[0] == input_act[0], "Dimensions doesn't match!!"
            B = output_act[0]
            output_act = output_act[1:]
            input_act = input_act[1:]
            weight = weight[-2:]
        if len(output_act) == 4:
            assert output_act[0] == input_act[0], "Dimensions doesn't match!!"
            assert output_act[1] == input_act[1], "Dimensions doesn't match!!"
            B = output_act[0] * output_act[1]
            output_act = output_act[2:]
            input_act = input_act[2:]
            weight = weight[-2:]

        N = output_act[0]
        M = output_act[1]
        P = 1
        Q = 1

        W = 1
        H = 1

        C = weight[0]
        R = 1
        S = 1

        Type = "CONV"

        W_stride = 1
        H_stride = 1
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        """
        if C <= definitions.TC_PE_x:
            C = C
        else:
            C = math.ceil(C/definitions.TC_PE_x) * definitions.TC_PE_x
        
        if M <= definitions.TC_PE_y:
            M = M
        else:
            M = math.ceil(M/definitions.TC_PE_y) * definitions.TC_PE_y
        """

        assert (
            output_act[0] == input_act[0]
            and output_act[1] == weight[1]
            and input_act[1] == weight[0]
        ), "Dimensions doesn't match!!" + str((output_act, input_act, weight))

    elif node.node_desc == "Bmm" or node.node_desc == "fused ~~ Bmm":
        output_act = node.output_act[0]
        if len(node.saved_tensors) > 2:
            input_act = node.saved_tensors[1]
            weight = node.saved_tensors[2]
        elif len(node.saved_tensors) == 2:
            input_act = node.saved_tensors[0]
            weight = node.saved_tensors[1]

        if len(output_act) == 3:
            assert output_act[0] == input_act[0] == weight[0], "Dimensions doesn't match!!"
            B = output_act[0]
            output_act = output_act[1:]
            input_act = input_act[1:]
            weight = weight[-2:]
        if len(output_act) == 4:
            assert output_act[0] == input_act[0] == weight[0], "Dimensions doesn't match!!"
            assert output_act[1] == input_act[1] == weight[0], "Dimensions doesn't match!!"
            B = output_act[0] * output_act[1]
            output_act = output_act[2:]
            input_act = input_act[2:]
            weight = weight[-2:]

        N = output_act[0]
        M = output_act[1]
        P = 1
        Q = 1

        W = 1
        H = 1

        C = weight[1]
        R = 1
        S = 1

        W_stride = 1
        H_stride = 1
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        Type = "CONV"

        assert (
            output_act[0] == input_act[0]
            and output_act[1] == weight[1]
            and input_act[1] == weight[0]
        ), "Dimensions doesn't match!!"

    return (
        B,
        N,
        M,
        C,
        W,
        H,
        R,
        S,
        W_stride,
        H_stride,
        W_dilation,
        H_dilation,
        Type,
        P,
        Q,
        W_padding,
        H_padding,
    )


def get_dims_and_fused_perf_est(schd, node,):
    dim_fwd = get_problem_dim(schd, node)

    fused = False
    fused_ops = []

    # Check if fused operator
    if node.node_desc.split(" ~~ ")[0] == "fused":
        fused = True
        fused_info = node.fused_operators.strip().split(" ~~ ")
        for i in range(len(fused_info)):
            if i % 3 == 0:
                name = fused_info[i]
            elif i % 3 == 1:
                weights = literal_eval(fused_info[i])
            elif i % 3 == 2:
                output = literal_eval(fused_info[i])
                op = {}
                op["name"] = name
                op["weights"] = weights
                op["output"] = output
                fused_ops.append(op)

    # Forward Path Estimation
    """
        https://arxiv.org/pdf/1603.07285.pdf

        Forward pass
        x = op1 = ip acts (N, C, X, Y) -> (1, 1024, 1, 1)
        W = op2 = weights (C, K, R, S) -> (1024, 1000, 1, 1)
        y = dest = op acts (N, K, Xo, Yo) -> (1, 1000, 1, 1)

        Conv -> x * W = y
        FC   -> x . W = y

        No zero padding, unit strides
        Xo = (X - R) + 1

        Zero padding, unit strides
        Xo = (X - R) + 2*W_pad + 1

        No zero padding, non-unit strides
        Xo = ((X - R) / W_stride) + 1

        Zero padding, non-unit stride
        Xo = ((X + 2*W_pad - R) / W_stride) + 1
    """

    (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = copy.deepcopy(dim_fwd)
    w_org = w
    h_org = h

    # sanity check
    if (
        node.node_desc == "CudnnConvolution"
        or node.node_desc == "MkldnnConvolution"
        or node.node_desc == "fused ~~ MkldnnConvolution"
        or node.node_desc == "fused ~~ CudnnConvolution"
    ):
        p_cal = math.floor((w + 2 * w_padding - r) / w_stride) + 1
        q_cal = math.floor((h + 2 * h_padding - s) / h_stride) + 1

        assert (
            p_cal == p and q_cal == q
        ), "forward pass input and output activations doesn't match!!"

        w = w + 2 * w_padding
        h = h + 2 * h_padding

    dim_fwd = (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    )

    dim_fwd = (
        b,
        n,
        m,
        c,
        w_org,
        h_org,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    )

    # Backward Path Estimation
    """
        Forward pass
        x = op1 = ip acts (N, C, X, Y) -> (1, 1024, 1, 1)
        W = op2 = weights (C, K, R, S) -> (1024, 1000, 1, 1)
        y = dest = op acts (N, K, Xo, Yo) -> (1, 1000, 1, 1)

        Backward pass
        L = loss
        dL/dy = op1 = (N, C, X, Y) -> (1, 1000, 1, 1)
        W(transpose) = op2 = (C, K, R, S) -> (1000, 1024, 1, 1)
        dL/dx = dest = (N, K, Xo, Yo) -> (1, 1024, 1, 1)

        Conv -> dL/dy * W(180 rotation) = dL/dx
        FC   -> dL/dy . W(transpose) = dL/dx

        op1 and dest gets swapped

        also check if activation sizes are different (op > ip) then
        we need to use transposed conv instead of normal conv

        if ip and op act size are same then we use normal CONV operation with padding on the op act

        if op act > ip act size then we use Transposed CONV operation as below

        No zero padding, unit strides
        Xo = X + (R - 1) --> W_pad = R - 1

        Zero padding, unit strides
        Xo = X + (R - 1) - 2*W_pad --> W_pad = R - W_pad - 1

        No zero padding, non-unit strides
        Xo = W_stride * (X - 1) + R --> W_pad = r - 1 and X' = X + ((X - 1) * (W_stride - 1)) and W_stride = 1

        Zero padding, non-unit stride
        Xo = W_stride * (X - 1) + R - 2*W_pad --> W_pad = R - W_pad - 1 and X' = X + ((X - 1) * (W_stride - 1)) and W_stride = 1    
    """
    (
        B,
        N,
        M,
        C,
        W,
        H,
        R,
        S,
        W_stride,
        H_stride,
        W_dilation,
        H_dilation,
        Type,
        P,
        Q,
        W_padding,
        H_padding,
    ) = copy.deepcopy(dim_fwd)
    (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = copy.deepcopy(dim_fwd)
    # factor a to be added to input and output
    a = (w + 2 * w_padding - r) % w_stride
    (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = (
        B,
        N,
        C,
        M,
        P,
        Q,
        R,
        S,
        W_stride,
        H_stride,
        W_dilation,
        H_dilation,
        Type,
        W,
        H,
        W_padding,
        H_padding,
    )

    # Check if activation sizes are different
    if p > w:
        if w_padding == 0 and w_stride == 1:
            w_padding = r - 1
            h_padding = s - 1
        elif w_padding == 0 and w_stride > 1:
            w_padding = r - 1
            h_padding = s - 1
            w = w + ((w - 1) * (w_stride - 1))
            h = h + ((h - 1) * (h_stride - 1))
            w_stride = 1
            h_stride = 1
        elif w_padding != 0 and w_stride == 1:
            w_padding = r - w_padding - 1
            h_padding = s - h_padding - 1
        elif w_padding != 0 and w_stride > 1:
            w_padding = r - w_padding - 1
            h_padding = s - h_padding - 1
            w = w + ((w - 1) * (w_stride - 1))
            h = h + ((h - 1) * (h_stride - 1))
            w_stride = 1
            h_stride = 1

    # sanity check
    if (
        node.node_desc == "CudnnConvolution"
        or node.node_desc == "MkldnnConvolution"
        or node.node_desc == "fused ~~ MkldnnConvolution"
        or node.node_desc == "fused ~~ CudnnConvolution"
    ):
        p_cal = math.floor((w + 2 * w_padding - r) / w_stride) + 1 + a
        q_cal = math.floor((h + 2 * h_padding - s) / h_stride) + 1 + a

        assert (
            p_cal == p and q_cal == q
        ), "backward pass input and output activations doesn't match!!"

        w = w + 2 * w_padding + a
        h = h + 2 * h_padding + a

    dim_bwd = (
        b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    )

    # # Weight Update Estimation
    # """
    #     Forward pass
    #     x = op1 = ip acts (N, C, X, Y) -> (1, 1024, 1, 1)
    #     W = op2 = weights (C, K, R, S) -> (1024, 1000, 1, 1)
    #     y = dest = op acts (N, K, Xo, Yo) -> (1, 1000, 1, 1)

    #     Backward pass
    #     L = loss
    #     dL/dy = op1 = (N, C, X, Y) -> (1, 1000, 1, 1)
    #     W(transpose) = op2 = (C, K, R, S) -> (1000, 1024, 1, 1)
    #     dL/dx = dest = (N, K, Xo, Yo) -> (1, 1024, 1, 1)

    #     Weight Update pass
    #     x(transpose) = op1 = (N, C, X, Y) -> (1024, 1, 1, 1)
    #     dL/dy = op2 = (C, K, R, S) -> (1, 1000, 1, 1)
    #     dL/dW = dest = (N, K, Xo, Yo) -> (1024, 1000, 1, 1)

    #     Conv -> x(transpose) * dL/dy = dL/dW
    #     FC   -> x(transpose) . dL/dy = dL/dW

    #     N and C gets swapped
    #     R,S get swapped with Xo,Yo
    #     stride needs to be updated as per Xo = [(X + 2*W_pad - R) / W_stride] + 1 if Conv operation

    # """
    # (
    #     B,
    #     N,
    #     M,
    #     C,
    #     W,
    #     H,
    #     R,
    #     S,
    #     W_stride,
    #     H_stride,
    #     W_dilation,
    #     H_dilation,
    #     Type,
    #     P,
    #     Q,
    #     W_padding,
    #     H_padding,
    # ) = copy.deepcopy(dim_fwd)
    # (
    #     b,
    #     n,
    #     m,
    #     c,
    #     w,
    #     h,
    #     r,
    #     s,
    #     w_stride,
    #     h_stride,
    #     w_dilation,
    #     h_dilation,
    #     types,
    #     p,
    #     q,
    #     w_padding,
    #     h_padding,
    # ) = copy.deepcopy(dim_fwd)
    # (
    #     B,
    #     N,
    #     M,
    #     C,
    #     W,
    #     H,
    #     R,
    #     S,
    #     W_stride,
    #     H_stride,
    #     W_dilation,
    #     H_dilation,
    #     Type,
    #     P,
    #     W,
    #     W_padding,
    #     H_padding,
    # ) = (
    #     b,
    #     c,
    #     m,
    #     n,
    #     w,
    #     h,
    #     p,
    #     q,
    #     w_stride,
    #     h_stride,
    #     w_dilation,
    #     h_dilation,
    #     types,
    #     r,
    #     s,
    #     w_padding,
    #     h_padding,
    # )

    # if (
    #     node.node_desc == "MkldnnConvolution"
    #     or node.node_desc == "CudnnConvolution"
    #     or node.node_desc == "ThnnConv2D"
    #     or node.node_desc == "CudnnRnn"
    #     or node.node_desc == "fused ~~ MkldnnConvolution"
    #     or node.node_desc == "fused ~~ CudnnConvolution"
    #     or node.node_desc == "fused ~~ ThnnConv2D"
    #     or node.node_desc == "fused ~~ CudnnRnn"
    # ):

    #     # In case Conv output is 1x1 then filter dimesnions should be equal to input act dimensions
    #     if P == 1:
    #         R = W
    #         S = H

    #     if W > R:
    #         W_stride = math.floor((W + 2 * W_padding - R) / (P - 1))
    #         H_stride = W_stride
    #     elif W == R:
    #         W_stride = W_stride
    #         H_stride = H_stride

    # # sanity check
    # if (
    #     node.node_desc == "CudnnConvolution"
    #     or node.node_desc == "MkldnnConvolution"
    #     or node.node_desc == "fused ~~ MkldnnConvolution"
    #     or node.node_desc == "fused ~~ CudnnConvolution"
    # ):
    #     p_cal = math.floor((W + 2 * W_padding - R) / W_stride) + 1
    #     q_cal = math.floor((H + 2 * H_padding - S) / H_stride) + 1

    #     # assert Xo_cal == Xo and \
    #     #        Yo_cal == Yo, \
    #     #        "weight update pass input and output activations doesn't match!!"

    #     W = W + 2 * W_padding
    #     H = H + 2 * H_padding

    # dim_wu = (
    #     N,
    #     N,
    #     M,
    #     C,
    #     W,
    #     H,
    #     R,
    #     S,
    #     W_stride,
    #     H_stride,
    #     W_dilation,
    #     H_dilation,
    #     Type,
    #     P,
    #     Q,
    #     W_padding,
    #     H_padding,
    # )

    fwd_PE_cycles = 0
    bwd_PE_cycles = 0

    if fused:
        num_ops = 0
        num_weights = 0
        for i in range(len(fused_ops)):
            op = fused_ops[i]
            if op_to_compute.get_opr_type(op["name"]) == "transformation_opr":
                continue
            else:
                fwd_PE_cycles += op_to_compute.get_fwd_pe_cycles(op["name"])
                bwd_PE_cycles += op_to_compute.get_bwd_pe_cycles(op["name"])
                for j in range(len(op["weights"])):
                    num_weights += op["weights"][j][0]
                num_ops += 1

        output = fused_ops[0]["output"][0]

        node.fwd_latency += fwd_PE_cycles
        node.bwd_latency += bwd_PE_cycles

    return [dim_fwd, dim_bwd], [fwd_PE_cycles, bwd_PE_cycles]
