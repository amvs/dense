import torch
def compute_out_groups(self):
    """
    all_out_groups[ind] = list of
        (depth, start_channel, end_channel)
    """
    K = self.n_orient
    dmax = self.depth
    J = self.n_scale
    C0 = self.in_channels

    # inputs: list of (depth, channels)
    inputs_groups = [(0, C0)]  # img first
    inputs_len = [C0]
    all_out_groups = [[(0, 0, C0)]]  # first layer: only img

    for ind in range(J):
        out_groups = []
        ch_ptr = 0

        for dep, ch in inputs_groups:
            new_dep = dep + 1
            new_ch = ch * K
            out_groups.append((new_dep, ch_ptr, ch_ptr + new_ch))
            ch_ptr += new_ch

        all_out_groups.append(out_groups)

        # build next inputs
        next_inputs = []

        for new_dep, start, end in out_groups:
            if new_dep <= dmax - 1:
                next_inputs.append((new_dep, end - start))

        inputs_groups.extend(next_inputs)
        print(f"Layer {ind+1} inputs_groups:", inputs_groups)
        inputs_len.append(sum([ch for _, ch in inputs_groups]))
    return all_out_groups, inputs_len

def build_keep_idx(self):
    all_out_groups, inputs_len = compute_out_groups(self)
    print("all_out_groups:", all_out_groups)
    d = self.depth

    keep_idx = []
    for groups in all_out_groups:
        idx = []
        for dep, start, end in groups:
            if dep <= d - 1:
                idx.extend(range(start, end))
        keep_idx.append(torch.tensor(idx, dtype=torch.long))
    return keep_idx, inputs_len


class Params:
    def __init__(self, n_orient, depth, n_scale, in_channels=1):
        self.n_orient = n_orient
        self.in_channels = in_channels
        self.depth = depth
        self.n_scale = n_scale


# Example usage
params = Params(n_orient=6, n_scale=7, depth=2)
keep_idx, keep_len = build_keep_idx(params)
for i, idx in enumerate(zip(keep_idx, keep_len)):
    print(f"Layer {i}: keep indices {idx[0].tolist()}, length {idx[1]}")